"""
ADAPTIVE Hi-LSTM v2 — Nifty 50 Multi-Timeframe Prediction Model
New Architecture:
  4 Bi-LSTM Encoders (15m, 1h, 1d, 1w)
  → Per-encoder Multi-Head Self-Attention
  → Hierarchical Cross-TF Gating (weekly gates daily, daily gates hourly, hourly gates 15m)
  → Regime Classifier (5 regimes)
  → Regime-Conditioned Decoder
  → 3 Output Heads: direction (3-class), magnitude (regression), confidence (scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


# ---------------------------------------------------------------------------
# Utility: weight initialisation
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module) -> None:
    """Apply Xavier (linear) and orthogonal (LSTM) initialisation recursively."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Initialise forget-gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# 1. TemporalEncoder
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """
    Encodes a single timeframe using Bi-LSTM + Multi-Head Self-Attention.

    Args:
        input_size  : number of features per timestep
        hidden_size : LSTM hidden size (output dim will be hidden_size * 2 due to bidirectional)
        num_layers  : LSTM depth
        n_heads     : number of attention heads
        dropout     : dropout rate

    Returns (via forward):
        context  : (batch, hidden_size * 2)  — pooled summary context vector
        sequence : (batch, seq_len, hidden_size * 2) — attention-refined full sequence
        attn_weights : (batch, seq_len, seq_len) — self-attention weights (last head avg)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_dim = hidden_size * 2  # bidirectional

        # Input projection — normalise raw features before feeding LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Multi-head self-attention over LSTM outputs
        # embed_dim must be divisible by n_heads
        attn_dim = self.out_dim
        if attn_dim % n_heads != 0:
            # Pad to nearest multiple
            attn_dim = ((attn_dim // n_heads) + 1) * n_heads
            self.attn_proj_in = nn.Linear(self.out_dim, attn_dim)
            self.attn_proj_out = nn.Linear(attn_dim, self.out_dim)
        else:
            self.attn_proj_in = None
            self.attn_proj_out = None

        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(self.out_dim)
        self.norm2 = nn.LayerNorm(self.out_dim)

        # Position-wise feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

        # Gradient clipping hook — applied during backward
        self._register_gradient_hooks()

        self.apply(_init_weights)

    # ------------------------------------------------------------------
    def _register_gradient_hooks(self) -> None:
        """Register backward hooks that clip per-parameter gradients to ±5."""
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5.0, 5.0))

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (batch, seq_len, input_size)
        Returns:
            context      : (batch, out_dim)
            sequence     : (batch, seq_len, out_dim)
            attn_weights : (batch, seq_len, seq_len)
        """
        # Project input
        x_proj = self.input_proj(x)  # (B, T, hidden_size)

        # Bi-LSTM
        lstm_out, (h_n, _) = self.lstm(x_proj)
        # h_n: (num_layers*2, B, hidden_size) — grab last layer fwd+bwd
        h_fwd = h_n[-2]  # (B, hidden_size)
        h_bwd = h_n[-1]  # (B, hidden_size)
        context_raw = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, out_dim)

        # Optional projection into attention dimension
        if self.attn_proj_in is not None:
            attn_in = self.attn_proj_in(lstm_out)
        else:
            attn_in = lstm_out  # (B, T, attn_dim)

        # Multi-head self-attention (residual)
        attn_out, attn_weights = self.self_attn(attn_in, attn_in, attn_in)

        if self.attn_proj_out is not None:
            attn_out = self.attn_proj_out(attn_out)

        # Residual + LayerNorm
        seq = self.norm1(lstm_out + self.dropout(attn_out))  # (B, T, out_dim)

        # FFN with residual
        seq = self.norm2(seq + self.ffn(seq))  # (B, T, out_dim)

        # Mean-pool over time for the context vector (more stable than last-step)
        context = seq.mean(dim=1)  # (B, out_dim)

        return context, seq, attn_weights


# ---------------------------------------------------------------------------
# 2. HierarchicalGate
# ---------------------------------------------------------------------------

class HierarchicalGate(nn.Module):
    """
    Gates the child encoder output using the parent encoder context.

    Implements:
        gate       = sigmoid( W * concat(child_ctx, parent_ctx) + b )
        gated_out  = child_ctx * gate

    The gate is then added back (residual) so the child signal is never fully
    suppressed:
        output = child_ctx + gated_out  (then LayerNorm)

    Args:
        child_size  : dimensionality of child context vector
        parent_size : dimensionality of parent context vector

    Returns:
        gated_context : (batch, child_size)
    """

    def __init__(self, child_size: int, parent_size: int) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(child_size + parent_size, child_size),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(child_size)
        self.apply(_init_weights)

        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5.0, 5.0))

    def forward(self, child_ctx: torch.Tensor, parent_ctx: torch.Tensor) -> torch.Tensor:
        """
        child_ctx  : (batch, child_size)
        parent_ctx : (batch, parent_size)
        Returns    : (batch, child_size)
        """
        combined = torch.cat([child_ctx, parent_ctx], dim=-1)
        gate = self.gate_net(combined)                   # (B, child_size)
        gated = child_ctx * gate
        return self.norm(child_ctx + gated)              # residual + norm


# ---------------------------------------------------------------------------
# 3. CrossTimeframeAttention
# ---------------------------------------------------------------------------

class CrossTimeframeAttention(nn.Module):
    """
    Cross-attention where the *child* timeframe provides queries and the
    *parent* timeframe provides keys and values.

    This lets a finer-grained encoder (e.g. 15m) selectively attend to the
    richer context of a coarser encoder (e.g. 1h), extracting the most
    relevant historical patterns.

    Args:
        child_dim  : output dimension of the child encoder (bidirectional → *2)
        parent_dim : output dimension of the parent encoder
        n_heads    : number of attention heads
        dropout    : dropout rate

    Returns:
        attended_context : (batch, child_dim) — attended representation
        cross_attn_weights : (batch, child_seq_len, parent_seq_len)
    """

    def __init__(
        self,
        child_dim: int,
        parent_dim: int,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Unify dimensions so Q and KV live in the same space
        self.unified_dim = child_dim
        if self.unified_dim % n_heads != 0:
            self.unified_dim = ((self.unified_dim // n_heads) + 1) * n_heads

        self.query_proj = nn.Linear(child_dim, self.unified_dim)
        self.key_proj   = nn.Linear(parent_dim, self.unified_dim)
        self.value_proj = nn.Linear(parent_dim, self.unified_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.unified_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Linear(self.unified_dim, child_dim)
        self.norm = nn.LayerNorm(child_dim)
        self.dropout = nn.Dropout(dropout)

        self.apply(_init_weights)
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5.0, 5.0))

    def forward(
        self,
        child_seq: torch.Tensor,
        parent_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        child_seq  : (batch, T_child, child_dim)
        parent_seq : (batch, T_parent, parent_dim)
        Returns:
            attended_context   : (batch, child_dim) — mean-pooled over child time
            cross_attn_weights : (batch, T_child, T_parent)
        """
        Q = self.query_proj(child_seq)   # (B, T_c, U)
        K = self.key_proj(parent_seq)    # (B, T_p, U)
        V = self.value_proj(parent_seq)  # (B, T_p, U)

        attn_out, weights = self.cross_attn(Q, K, V)  # (B, T_c, U), (B, T_c, T_p)

        attn_out = self.out_proj(attn_out)             # (B, T_c, child_dim)

        # Residual + norm over sequence
        attended_seq = self.norm(child_seq + self.dropout(attn_out))

        # Pool to produce a single context vector
        attended_context = attended_seq.mean(dim=1)    # (B, child_dim)

        return attended_context, weights


# ---------------------------------------------------------------------------
# 4. RegimeClassifier
# ---------------------------------------------------------------------------

class RegimeClassifier(nn.Module):
    """
    Classifies current market regime from the merged multi-timeframe context.

    Input  : merged context (batch, merged_dim)
    Output : regime_probs (batch, n_regimes) via softmax
             Regimes: 0=strong_bull, 1=weak_bull, 2=ranging,
                      3=weak_bear,   4=strong_bear

    Architecture: 2-layer MLP with BatchNorm and dropout for robustness.
    """

    def __init__(self, merged_dim: int, n_regimes: int = 5, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_regimes = n_regimes

        self.net = nn.Sequential(
            nn.Linear(merged_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_regimes),
        )

        self.apply(_init_weights)
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5.0, 5.0))

    def forward(self, merged_ctx: torch.Tensor) -> torch.Tensor:
        """
        merged_ctx : (batch, merged_dim)
        Returns    : (batch, n_regimes) — soft probabilities
        """
        logits = self.net(merged_ctx)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# 5. RegimeConditionedDecoder
# ---------------------------------------------------------------------------

class RegimeConditionedDecoder(nn.Module):
    """
    Five parallel decoder branches (one per regime).
    Final prediction = weighted sum of branch outputs using regime_probs.

    Each branch:
        Linear(merged_dim, 256) → GELU → Dropout
        → Linear(256, 128) → GELU → Dropout
        → shared output heads

    Output heads:
        direction_logits : (batch, 3)  — up / flat / down  [CrossEntropy]
        magnitude        : (batch, 1)  — % change           [MSE]
        confidence       : (batch, 1)  — 0-1 score          [BCE / free]

    Args:
        merged_dim : total dimension of the merged context
        n_regimes  : number of parallel decoder branches
        dropout    : dropout rate
    """

    def __init__(
        self,
        merged_dim: int,
        n_regimes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_regimes = n_regimes

        # Build n_regimes parallel branch trunks as a ModuleList
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(merged_dim, 256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_regimes)
            ]
        )

        # Shared output heads (applied after each branch)
        self.dir_heads  = nn.ModuleList([nn.Linear(128, 3) for _ in range(n_regimes)])
        self.mag_heads  = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_regimes)])
        self.conf_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_regimes)])

        self.apply(_init_weights)
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5.0, 5.0))

    def forward(
        self,
        merged_ctx: torch.Tensor,
        regime_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        merged_ctx   : (batch, merged_dim)
        regime_probs : (batch, n_regimes)

        Returns:
            direction_logits : (batch, 3)
            magnitude        : (batch, 1)
            confidence       : (batch, 1)  — after sigmoid
        """
        batch = merged_ctx.size(0)

        # Accumulate weighted branch outputs
        dir_acc  = torch.zeros(batch, 3,  device=merged_ctx.device)
        mag_acc  = torch.zeros(batch, 1,  device=merged_ctx.device)
        conf_acc = torch.zeros(batch, 1,  device=merged_ctx.device)

        for i in range(self.n_regimes):
            w = regime_probs[:, i : i + 1]          # (B, 1) — regime weight

            feat = self.branches[i](merged_ctx)       # (B, 128)

            dir_acc  = dir_acc  + w * self.dir_heads[i](feat)
            mag_acc  = mag_acc  + w * self.mag_heads[i](feat)
            conf_acc = conf_acc + w * self.conf_heads[i](feat)

        confidence = torch.sigmoid(conf_acc)          # bound to (0, 1)

        return dir_acc, mag_acc, confidence


# ---------------------------------------------------------------------------
# 6. AdaptiveHiLSTMv2  (main model)
# ---------------------------------------------------------------------------

class AdaptiveHiLSTMv2(nn.Module):
    """
    Main model combining all ADAPTIVE Hi-LSTM v2 components.

    Forward inputs:
        x_15m : (batch, seq_15m, features_15m)
        x_1h  : (batch, seq_1h,  features_1h)
        x_1d  : (batch, seq_1d,  features_1d)
        x_1w  : (batch, seq_1w,  features_1w)

    Forward outputs (dict):
        direction_logits : (batch, 3)
        magnitude        : (batch, 1)
        confidence       : (batch, 1)
        regime_probs     : (batch, 5)

    Architecture flow:
        1. Each timeframe → TemporalEncoder → (context_vec, full_sequence)
        2. HierarchicalGate: gate 1d context with 1w context
        3. HierarchicalGate: gate 1h context with 1d (gated) context
        4. HierarchicalGate: gate 15m context with 1h (gated) context
        5. CrossTimeframeAttention: 15m attends to 1h seq
           CrossTimeframeAttention: 1h  attends to 1d seq
           CrossTimeframeAttention: 1d  attends to 1w seq
        6. Concatenate all gated + cross-attended contexts → merged
        7. RegimeClassifier(merged) → regime_probs
        8. RegimeConditionedDecoder(merged, regime_probs) → 3 heads
    """

    def __init__(
        self,
        features_15m: int,
        features_1h: int,
        features_1d: int,
        features_1w: int,
        hidden_15m: int = 256,
        hidden_1h: int = 192,
        hidden_1d: int = 128,
        hidden_1w: int = 64,
        layers_15m: int = 3,
        layers_1h: int = 2,
        layers_1d: int = 2,
        layers_1w: int = 2,
        n_heads: int = 8,
        dropout: float = 0.3,
        n_regimes: int = 5,
    ) -> None:
        super().__init__()

        # Output dims (bidirectional → *2)
        self.out_15m = hidden_15m * 2
        self.out_1h  = hidden_1h  * 2
        self.out_1d  = hidden_1d  * 2
        self.out_1w  = hidden_1w  * 2

        # ── Temporal Encoders ─────────────────────────────────────────────
        self.enc_15m = TemporalEncoder(features_15m, hidden_15m, layers_15m, n_heads, dropout)
        self.enc_1h  = TemporalEncoder(features_1h,  hidden_1h,  layers_1h,  n_heads, dropout)
        self.enc_1d  = TemporalEncoder(features_1d,  hidden_1d,  layers_1d,  n_heads, dropout)
        self.enc_1w  = TemporalEncoder(features_1w,  hidden_1w,  layers_1w,  n_heads, dropout)

        # ── Hierarchical Gates (parent gates child) ───────────────────────
        # 1w gates 1d
        self.gate_1d  = HierarchicalGate(self.out_1d, self.out_1w)
        # 1d gates 1h
        self.gate_1h  = HierarchicalGate(self.out_1h, self.out_1d)
        # 1h gates 15m
        self.gate_15m = HierarchicalGate(self.out_15m, self.out_1h)

        # ── Cross-Timeframe Attention ─────────────────────────────────────
        # 15m (child) attends to 1h (parent)
        self.cross_15m_1h = CrossTimeframeAttention(self.out_15m, self.out_1h, n_heads, dropout)
        # 1h (child) attends to 1d (parent)
        self.cross_1h_1d  = CrossTimeframeAttention(self.out_1h,  self.out_1d, n_heads, dropout)
        # 1d (child) attends to 1w (parent)
        self.cross_1d_1w  = CrossTimeframeAttention(self.out_1d,  self.out_1w, n_heads, dropout)

        # ── Merged dimension ─────────────────────────────────────────────
        # Each timeframe contributes: gated_ctx + cross_attended_ctx  (2× each)
        # 1w has no parent, so contributes just its context once
        self.merged_dim = (
            self.out_15m * 2   # gated + cross-attended
            + self.out_1h  * 2
            + self.out_1d  * 2
            + self.out_1w        # 1w context (no parent gate / cross-attn from above)
        )

        # ── Projection before regime classifier ──────────────────────────
        self.merge_proj = nn.Sequential(
            nn.Linear(self.merged_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_dim = 512

        # ── Regime Classifier & Decoder ──────────────────────────────────
        self.regime_clf = RegimeClassifier(self.proj_dim, n_regimes, dropout)
        self.decoder    = RegimeConditionedDecoder(self.proj_dim, n_regimes, dropout)

        # Final gradient-clip hooks on top-level params (encoders already have them)
        for p in self.merge_proj.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -5.0, 5.0))

        self.n_regimes = n_regimes
        self.dropout_rate = dropout

        # ── Multi-horizon auxiliary heads ───────────────────────────────
        # Exposes 15m / 1h / 1d views for inspection and optional auxiliary training.
        self.horizon_names = ("15m", "1h", "1d")
        self.horizon_adapters = nn.ModuleDict(
            {
                h: nn.Sequential(
                    nn.Linear(self.proj_dim, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for h in self.horizon_names
            }
        )
        self.horizon_dir_heads = nn.ModuleDict({h: nn.Linear(256, 3) for h in self.horizon_names})
        self.horizon_mag_heads = nn.ModuleDict({h: nn.Linear(256, 1) for h in self.horizon_names})
        self.horizon_conf_heads = nn.ModuleDict({h: nn.Linear(256, 1) for h in self.horizon_names})

    # ------------------------------------------------------------------
    def _encode_all(
        self, x_15m, x_1h, x_1d, x_1w
    ):
        """Run all four encoders and return contexts + sequences."""
        ctx_15m, seq_15m, aw_15m = self.enc_15m(x_15m)
        ctx_1h,  seq_1h,  aw_1h  = self.enc_1h(x_1h)
        ctx_1d,  seq_1d,  aw_1d  = self.enc_1d(x_1d)
        ctx_1w,  seq_1w,  aw_1w  = self.enc_1w(x_1w)
        return (
            ctx_15m, seq_15m, aw_15m,
            ctx_1h,  seq_1h,  aw_1h,
            ctx_1d,  seq_1d,  aw_1d,
            ctx_1w,  seq_1w,  aw_1w,
        )

    # ------------------------------------------------------------------
    def _build_merged(self, ctx_15m, seq_15m, ctx_1h, seq_1h, ctx_1d, seq_1d, ctx_1w, seq_1w):
        """Apply gates, cross-attention, and project to fixed dim."""
        # ── Hierarchical gating ──────────────────────────────────────
        gated_1d  = self.gate_1d(ctx_1d,  ctx_1w)
        gated_1h  = self.gate_1h(ctx_1h,  gated_1d)
        gated_15m = self.gate_15m(ctx_15m, gated_1h)

        # ── Cross-timeframe attention ────────────────────────────────
        cross_15m, caw_15m = self.cross_15m_1h(seq_15m, seq_1h)
        cross_1h,  caw_1h  = self.cross_1h_1d(seq_1h,  seq_1d)
        cross_1d,  caw_1d  = self.cross_1d_1w(seq_1d,  seq_1w)

        # ── Concatenate all representations ─────────────────────────
        merged = torch.cat(
            [
                gated_15m, cross_15m,
                gated_1h,  cross_1h,
                gated_1d,  cross_1d,
                ctx_1w,
            ],
            dim=-1,
        )  # (B, merged_dim)

        merged_proj = self.merge_proj(merged)  # (B, proj_dim)

        cross_attn_weights = {
            "15m_to_1h": caw_15m,
            "1h_to_1d":  caw_1h,
            "1d_to_1w":  caw_1d,
        }

        return merged_proj, cross_attn_weights

    # ------------------------------------------------------------------
    def forward(
        self,
        x_15m: torch.Tensor,
        x_1h:  torch.Tensor,
        x_1d:  torch.Tensor,
        x_1w:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys:
            direction_logits, magnitude, confidence, regime_probs
        """
        (
            ctx_15m, seq_15m, _,
            ctx_1h,  seq_1h,  _,
            ctx_1d,  seq_1d,  _,
            ctx_1w,  seq_1w,  _,
        ) = self._encode_all(x_15m, x_1h, x_1d, x_1w)

        merged_proj, _ = self._build_merged(
            ctx_15m, seq_15m, ctx_1h, seq_1h, ctx_1d, seq_1d, ctx_1w, seq_1w
        )

        regime_probs = self.regime_clf(merged_proj)

        direction_logits, magnitude, confidence = self.decoder(merged_proj, regime_probs)
        horizon_predictions = self._predict_horizons(merged_proj)

        return {
            "direction_logits": direction_logits,   # (B, 3)
            "magnitude":        magnitude,           # (B, 1)
            "confidence":       confidence,          # (B, 1)
            "regime_probs":     regime_probs,        # (B, 5)
            "horizon_predictions": horizon_predictions,
        }


    # ------------------------------------------------------------------
    def _predict_horizons(self, merged_proj: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return auxiliary 15m / 1h / 1d horizon heads for visualization."""
        horizon_predictions: Dict[str, Dict[str, torch.Tensor]] = {}
        for horizon in self.horizon_names:
            feat = self.horizon_adapters[horizon](merged_proj)
            horizon_predictions[horizon] = {
                "direction_logits": self.horizon_dir_heads[horizon](feat),
                "magnitude": self.horizon_mag_heads[horizon](feat),
                "confidence": torch.sigmoid(self.horizon_conf_heads[horizon](feat)),
            }
        return horizon_predictions

    # ------------------------------------------------------------------
    def predict_with_uncertainty(
        self,
        x_15m: torch.Tensor,
        x_1h:  torch.Tensor,
        x_1d:  torch.Tensor,
        x_1w:  torch.Tensor,
        n_passes: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo Dropout uncertainty estimation.

        Forces dropout ON (train mode) for n_passes stochastic forward passes,
        then returns mean predictions and standard deviation (uncertainty).

        Returns dict:
            direction_logits_mean : (B, 3)
            direction_logits_std  : (B, 3)
            magnitude_mean        : (B, 1)
            magnitude_std         : (B, 1)
            confidence_mean       : (B, 1)
            confidence_std        : (B, 1)
            regime_probs_mean     : (B, 5)
            regime_probs_std      : (B, 5)
        """
        was_training = self.training
        self.train()  # ensure dropout is active

        all_dir  = []
        all_mag  = []
        all_conf = []
        all_reg  = []
        all_hor  = []

        with torch.no_grad():
            for _ in range(n_passes):
                out = self.forward(x_15m, x_1h, x_1d, x_1w)
                all_dir.append(out["direction_logits"].unsqueeze(0))
                all_mag.append(out["magnitude"].unsqueeze(0))
                all_conf.append(out["confidence"].unsqueeze(0))
                all_reg.append(out["regime_probs"].unsqueeze(0))
                all_hor.append(out.get("horizon_predictions", {}))

        # Stack: (n_passes, B, *)
        dir_stack  = torch.cat(all_dir,  dim=0)
        mag_stack  = torch.cat(all_mag,  dim=0)
        conf_stack = torch.cat(all_conf, dim=0)
        reg_stack  = torch.cat(all_reg,  dim=0)

        horizon_mean: Dict[str, Dict[str, torch.Tensor]] = {}
        if all_hor and all_hor[0]:
            for horizon in self.horizon_names:
                dir_list, mag_list, conf_list = [], [], []
                for sample in all_hor:
                    if horizon not in sample:
                        continue
                    dir_list.append(sample[horizon]["direction_logits"].unsqueeze(0))
                    mag_list.append(sample[horizon]["magnitude"].unsqueeze(0))
                    conf_list.append(sample[horizon]["confidence"].unsqueeze(0))
                if dir_list:
                    d_stack = torch.cat(dir_list, 0)
                    m_stack = torch.cat(mag_list, 0)
                    c_stack = torch.cat(conf_list, 0)
                    horizon_mean[horizon] = {
                        "direction_logits_mean": d_stack.mean(0),
                        "direction_logits_std": d_stack.std(0),
                        "magnitude_mean": m_stack.mean(0),
                        "magnitude_std": m_stack.std(0),
                        "confidence_mean": c_stack.mean(0),
                        "confidence_std": c_stack.std(0),
                    }

        if not was_training:
            self.eval()

        return {
            "direction_logits_mean": dir_stack.mean(0),
            "direction_logits_std":  dir_stack.std(0),
            "magnitude_mean":        mag_stack.mean(0),
            "magnitude_std":         mag_stack.std(0),
            "confidence_mean":       conf_stack.mean(0),
            "confidence_std":        conf_stack.std(0),
            "regime_probs_mean":     reg_stack.mean(0),
            "regime_probs_std":      reg_stack.std(0),
            "horizon_predictions_mean": horizon_mean,
        }

    # ------------------------------------------------------------------
    def get_attention_weights(
        self,
        x_15m: torch.Tensor,
        x_1h:  torch.Tensor,
        x_1d:  torch.Tensor,
        x_1w:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns all attention weights for interpretability / visualisation.

        Keys:
            self_attn_15m       : (B, T_15m, T_15m) — self-attention inside 15m encoder
            self_attn_1h        : (B, T_1h, T_1h)
            self_attn_1d        : (B, T_1d, T_1d)
            self_attn_1w        : (B, T_1w, T_1w)
            cross_15m_to_1h     : (B, T_15m, T_1h)
            cross_1h_to_1d      : (B, T_1h,  T_1d)
            cross_1d_to_1w      : (B, T_1d,  T_1w)
        """
        with torch.no_grad():
            ctx_15m, seq_15m, aw_15m, \
            ctx_1h,  seq_1h,  aw_1h,  \
            ctx_1d,  seq_1d,  aw_1d,  \
            ctx_1w,  seq_1w,  aw_1w = self._encode_all(x_15m, x_1h, x_1d, x_1w)

            _, cross_attn_weights = self._build_merged(
                ctx_15m, seq_15m, ctx_1h, seq_1h, ctx_1d, seq_1d, ctx_1w, seq_1w
            )

        return {
            "self_attn_15m":   aw_15m,
            "self_attn_1h":    aw_1h,
            "self_attn_1d":    aw_1d,
            "self_attn_1w":    aw_1w,
            "cross_15m_to_1h": cross_attn_weights["15m_to_1h"],
            "cross_1h_to_1d":  cross_attn_weights["1h_to_1d"],
            "cross_1d_to_1w":  cross_attn_weights["1d_to_1w"],
        }

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 7. EnsembleModel
# ---------------------------------------------------------------------------

class EnsembleModel(nn.Module):
    """
    Ensemble of N AdaptiveHiLSTMv2 models (each trained with a different seed).

    Aggregation strategy:
        direction_logits : mean of softmax probabilities, then log for loss
        magnitude        : mean
        confidence       : mean
        regime_probs     : mean

    Additional output:
        ensemble_disagreement : scalar — mean std across models for direction probs,
                                used as a signal of model uncertainty.
    """

    def __init__(self, models: List[AdaptiveHiLSTMv2]) -> None:
        super().__init__()
        if len(models) == 0:
            raise ValueError("EnsembleModel requires at least one model.")
        self.models = nn.ModuleList(models)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_15m: torch.Tensor,
        x_1h:  torch.Tensor,
        x_1d:  torch.Tensor,
        x_1w:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns averaged predictions and ensemble disagreement.

        Keys:
            direction_logits    : (B, 3)  — averaged log-probabilities
            magnitude           : (B, 1)
            confidence          : (B, 1)
            regime_probs        : (B, 5)
            ensemble_disagreement : (B,) — std of direction probs across models
        """
        all_dir_probs = []
        all_mag       = []
        all_conf      = []
        all_reg       = []

        for model in self.models:
            out = model(x_15m, x_1h, x_1d, x_1w)
            all_dir_probs.append(F.softmax(out["direction_logits"], dim=-1).unsqueeze(0))
            all_mag.append(out["magnitude"].unsqueeze(0))
            all_conf.append(out["confidence"].unsqueeze(0))
            all_reg.append(out["regime_probs"].unsqueeze(0))

        # Stack along model dimension: (N, B, *)
        dir_stack  = torch.cat(all_dir_probs, dim=0)   # (N, B, 3)
        mag_stack  = torch.cat(all_mag,       dim=0)   # (N, B, 1)
        conf_stack = torch.cat(all_conf,      dim=0)   # (N, B, 1)
        reg_stack  = torch.cat(all_reg,       dim=0)   # (N, B, 5)

        mean_dir_probs = dir_stack.mean(0)              # (B, 3)
        mean_mag       = mag_stack.mean(0)
        mean_conf      = conf_stack.mean(0)
        mean_reg       = reg_stack.mean(0)

        # Disagreement: average std of direction probs across models
        dir_std = dir_stack.std(0)                      # (B, 3)
        disagreement = dir_std.mean(dim=-1)             # (B,)

        # Return log-probs so callers can apply NLLLoss or CrossEntropy
        mean_dir_logits = torch.log(mean_dir_probs + 1e-8)

        return {
            "direction_logits":      mean_dir_logits,
            "magnitude":             mean_mag,
            "confidence":            mean_conf,
            "regime_probs":          mean_reg,
            "ensemble_disagreement": disagreement,
        }

    # ------------------------------------------------------------------
    def predict_with_uncertainty(
        self,
        x_15m: torch.Tensor,
        x_1h:  torch.Tensor,
        x_1d:  torch.Tensor,
        x_1w:  torch.Tensor,
        n_passes: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Combines ensemble diversity with MC-Dropout per member for richer
        uncertainty estimates.

        Returns same keys as individual model's predict_with_uncertainty,
        plus `ensemble_disagreement`.
        """
        all_results = []
        for model in self.models:
            res = model.predict_with_uncertainty(x_15m, x_1h, x_1d, x_1w, n_passes)
            all_results.append(res)

        # Average means and combine stds (quadrature)
        keys_mean = [
            "direction_logits_mean", "magnitude_mean",
            "confidence_mean", "regime_probs_mean",
        ]
        keys_std = [
            "direction_logits_std", "magnitude_std",
            "confidence_std", "regime_probs_std",
        ]

        out: Dict[str, torch.Tensor] = {}
        for k in keys_mean:
            stacked = torch.stack([r[k] for r in all_results], dim=0)
            out[k] = stacked.mean(0)

        for k in keys_std:
            stacked = torch.stack([r[k] for r in all_results], dim=0)
            # Total uncertainty = sqrt(mean(var) + var(mean))
            out[k] = (stacked.pow(2).mean(0)
                      + torch.stack([r[k.replace("_std", "_mean")] for r in all_results]).var(0)
                      ).sqrt()

        # Disagreement from direction mean stds
        out["ensemble_disagreement"] = out["direction_logits_std"].mean(dim=-1)

        return out
