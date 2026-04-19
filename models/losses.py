"""
ADAPTIVE Hi-LSTM v2 — Custom Trading Loss Functions
====================================================

AdaptiveTradingLoss
-------------------
Combines five sub-losses into a single differentiable objective tailored for
financial time-series prediction:

  1. CrossEntropyLoss for direction classification   (weight: 1.0)
  2. MSELoss for magnitude regression                (weight: 0.3)
  3. Regime entropy regularisation                   (weight: 0.1)
     — penalises over-confident single-regime predictions;
       encourages the model to distribute probability mass across regimes
       when uncertainty is high, preventing collapse to one regime.
  4. Direction-magnitude consistency penalty         (weight: 0.2)
     — if predicted direction is UP (class 0) but predicted magnitude < 0,
       or direction is DOWN (class 2) but magnitude > 0, apply an extra
       squared penalty proportional to the inconsistency.
  5. Asymmetric strong-move penalty                  (weight: 0.2)
     — wrong calls on large actual moves (|y_magnitude| > threshold) are
       penalised 2x relative to weak moves.

Usage
-----
    criterion = AdaptiveTradingLoss()
    total_loss, components = criterion(
        direction_logits,   # (B, 3)
        magnitude,          # (B, 1)
        confidence,         # (B, 1)
        regime_probs,       # (B, 5)
        y_direction,        # (B,)   long tensor -- 0=up, 1=flat, 2=down
        y_magnitude,        # (B, 1) float tensor -- actual % change
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class AdaptiveTradingLoss(nn.Module):
    """
    Multi-component trading loss for AdaptiveHiLSTMv2.

    Args:
        dir_weight              : weight for direction CrossEntropy loss
        mag_weight              : weight for magnitude MSE loss
        regime_entropy_weight   : weight for regime entropy regularisation
        consistency_weight      : weight for direction-magnitude consistency loss
        asymmetric_weight       : weight for asymmetric strong-move penalty
        strong_move_threshold   : |y_magnitude| above this is a 'strong move'
        asymmetric_multiplier   : multiplier applied to errors on strong moves
        class_weights           : optional (3,) tensor for imbalanced direction classes
        eps                     : numerical stability epsilon
    """

    def __init__(
        self,
        dir_weight: float = 1.0,
        mag_weight: float = 0.3,
        regime_entropy_weight: float = 0.1,
        consistency_weight: float = 0.2,
        asymmetric_weight: float = 0.2,
        strong_move_threshold: float = 0.005,
        asymmetric_multiplier: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.dir_weight             = dir_weight
        self.mag_weight             = mag_weight
        self.regime_entropy_weight  = regime_entropy_weight
        self.consistency_weight     = consistency_weight
        self.asymmetric_weight      = asymmetric_weight
        self.strong_move_threshold  = strong_move_threshold
        self.asymmetric_multiplier  = asymmetric_multiplier
        self.eps                    = eps

        # Register class weights as buffer so they move with .to(device)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self.ce_loss  = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------ #
    # Sub-loss helpers                                                     #
    # ------------------------------------------------------------------ #

    def _direction_loss(
        self,
        direction_logits: torch.Tensor,   # (B, 3)
        y_direction: torch.Tensor,         # (B,)  long
    ) -> torch.Tensor:
        """Standard cross-entropy over 3 direction classes. Returns mean scalar."""
        return self.ce_loss(direction_logits, y_direction).mean()

    def _magnitude_loss(
        self,
        magnitude: torch.Tensor,    # (B, 1)
        y_magnitude: torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:
        """MSE between predicted and actual % change. Returns mean scalar.

        Flatten both tensors so (B, 1) and (B,) are treated identically and
        PyTorch does not emit a shape/broadcasting warning.
        """
        magnitude = magnitude.view(-1)
        y_magnitude = y_magnitude.view(-1)
        return self.mse_loss(magnitude, y_magnitude).mean()

    def _regime_entropy_loss(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Encourage diversity in regime predictions via entropy regularisation.

        Maximise entropy of the regime distribution (i.e. minimise
        negative entropy).  High entropy = model is uncertain across regimes.
        The loss penalises low-entropy (overconfident single-regime) predictions.

        Loss = -mean( H(regime_probs) ) / log(n_regimes)   in [-1, 0]

        Negative because we want to PENALISE low entropy (return a positive loss
        that can be added to the total).
        """
        n_regimes = regime_probs.size(-1)
        p = torch.clamp(regime_probs, min=self.eps)
        entropy = -(p * torch.log(p)).sum(dim=-1)          # (B,)
        max_entropy = torch.log(torch.tensor(float(n_regimes), device=p.device))
        normalised_entropy = entropy / max_entropy          # (B,) in [0, 1]
        # Return negative normalised entropy as penalty for LOW entropy
        return -normalised_entropy.mean()

    def _consistency_loss(
        self,
        direction_logits: torch.Tensor,  # (B, 3)
        magnitude: torch.Tensor,         # (B, 1)
    ) -> torch.Tensor:
        """
        Penalise contradictory direction/magnitude pairs using soft class probs:
            up_prob   * relu(-magnitude)   (predicted up, but magnitude negative)
            down_prob * relu(+magnitude)   (predicted down, but magnitude positive)
        Fully differentiable — no hard argmax.
        """
        probs     = F.softmax(direction_logits, dim=-1)
        up_prob   = probs[:, 0:1]   # (B, 1)
        down_prob = probs[:, 2:3]   # (B, 1)

        inconsistency = (
            up_prob   * F.relu(-magnitude)
            + down_prob * F.relu(magnitude)
        )
        return inconsistency.pow(2).mean()

    def _asymmetric_loss(
        self,
        direction_logits: torch.Tensor,   # (B, 3)
        magnitude: torch.Tensor,          # (B, 1)
        y_direction: torch.Tensor,        # (B,)  long
        y_magnitude: torch.Tensor,        # (B, 1)
    ) -> torch.Tensor:
        """
        Apply asymmetric_multiplier to CE loss for strong-move samples.

        Errors on large actual moves (|y_magnitude| > threshold) receive
        higher penalty, encouraging accuracy where it matters most financially.
        """
        per_sample_ce = self.ce_loss(direction_logits, y_direction)      # (B,)

        strong = (y_magnitude.abs() > self.strong_move_threshold).squeeze(-1).float()  # (B,)
        weight = 1.0 + (self.asymmetric_multiplier - 1.0) * strong       # (B,)

        return (weight * per_sample_ce).mean()

    # ------------------------------------------------------------------ #
    # Main forward                                                        #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        direction_logits: torch.Tensor,   # (B, 3)
        magnitude: torch.Tensor,          # (B, 1)
        confidence: torch.Tensor,         # (B, 1)  -- reserved
        regime_probs: torch.Tensor,       # (B, 5)
        y_direction: torch.Tensor,        # (B,)   long -- 0=up, 1=flat, 2=down
        y_magnitude: torch.Tensor,        # (B, 1) float -- actual % change
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss and return per-component breakdown.

        Returns:
            total_loss       : scalar tensor (fully differentiable)
            loss_components  : dict of individual loss values (detached scalars)
        """
        # 1. Direction cross-entropy
        l_dir     = self._direction_loss(direction_logits, y_direction)

        # 2. Magnitude MSE
        l_mag     = self._magnitude_loss(magnitude, y_magnitude)

        # 3. Regime entropy regularisation
        l_regime  = self._regime_entropy_loss(regime_probs)

        # 4. Direction-magnitude consistency
        l_consist = self._consistency_loss(direction_logits, magnitude)

        # 5. Asymmetric strong-move penalty
        l_asym    = self._asymmetric_loss(direction_logits, magnitude, y_direction, y_magnitude)

        # Weighted sum
        total_loss = (
            self.dir_weight              * l_dir
            + self.mag_weight            * l_mag
            + self.regime_entropy_weight * l_regime
            + self.consistency_weight    * l_consist
            + self.asymmetric_weight     * l_asym
        )

        loss_components: Dict[str, torch.Tensor] = {
            "total":               total_loss.detach(),
            "direction_ce":        l_dir.detach(),
            "magnitude_mse":       l_mag.detach(),
            "regime_entropy":      l_regime.detach(),
            "consistency":         l_consist.detach(),
            "asymmetric_strong":   l_asym.detach(),
            # Weighted contributions
            "w_direction_ce":      (self.dir_weight              * l_dir).detach(),
            "w_magnitude_mse":     (self.mag_weight              * l_mag).detach(),
            "w_regime_entropy":    (self.regime_entropy_weight   * l_regime).detach(),
            "w_consistency":       (self.consistency_weight      * l_consist).detach(),
            "w_asymmetric_strong": (self.asymmetric_weight       * l_asym).detach(),
        }

        return total_loss, loss_components


# ---------------------------------------------------------------------------
# ConfidenceCalibrationLoss  (auxiliary)
# ---------------------------------------------------------------------------

class ConfidenceCalibrationLoss(nn.Module):
    """
    Auxiliary loss to calibrate the confidence head.

    Aligns predicted confidence with the empirical accuracy of the direction
    head: if the model predicts confidence=0.9 it should be correct ~90% of
    the time.

    Loss = MSE( confidence, correctness )
    where correctness = 1 if argmax(direction_logits) == y_direction else 0.

    The correctness target is a stop-gradient so it does not interfere with
    the direction head's learning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        confidence: torch.Tensor,         # (B, 1)
        direction_logits: torch.Tensor,   # (B, 3)
        y_direction: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        with torch.no_grad():
            pred_class  = direction_logits.argmax(dim=-1)                    # (B,)
            correctness = (pred_class == y_direction).float().unsqueeze(-1)  # (B, 1)

        return self.mse(confidence, correctness)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_loss(
    dir_weight: float = 1.0,
    mag_weight: float = 0.3,
    regime_entropy_weight: float = 0.1,
    consistency_weight: float = 0.2,
    asymmetric_weight: float = 0.2,
    strong_move_threshold: float = 0.005,
    asymmetric_multiplier: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> AdaptiveTradingLoss:
    """
    Convenience factory that returns a configured AdaptiveTradingLoss.

    Example:
        criterion = build_loss(strong_move_threshold=0.01)
    """
    return AdaptiveTradingLoss(
        dir_weight=dir_weight,
        mag_weight=mag_weight,
        regime_entropy_weight=regime_entropy_weight,
        consistency_weight=consistency_weight,
        asymmetric_weight=asymmetric_weight,
        strong_move_threshold=strong_move_threshold,
        asymmetric_multiplier=asymmetric_multiplier,
        class_weights=class_weights,
    )
