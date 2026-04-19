from __future__ import annotations

import argparse
import csv
from pathlib import Path

from config import Config
from learning.daily_feedback import DailyFeedbackLearner


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply daily outcome feedback to the RL-style tuner.")
    parser.add_argument("--predictions", required=True, help="CSV path with prediction rows from the live log")
    parser.add_argument("--outcomes", required=True, help="CSV path with columns: timestamp,actual_return[,realized_pnl]")
    parser.add_argument("--state", default=Config.RL_STATE_FILE, help="Path to the feedback state JSON")
    args = parser.parse_args()

    learner = DailyFeedbackLearner(args.state, history_limit=getattr(Config, "FEEDBACK_HISTORY_LIMIT", 500))

    outcomes = {}
    with open(args.outcomes, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            outcomes[row["timestamp"]] = row

    updated = 0
    with open(args.predictions, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")
            if ts not in outcomes:
                continue
            out = outcomes[ts]
            actual = float(out.get("actual_return", 0.0))
            pnl = out.get("realized_pnl")
            pnl_val = float(pnl) if pnl not in (None, "") else None
            try:
                learner.update_outcome(ts, actual_return=actual, realized_pnl=pnl_val, regime=row.get("regime", "Unknown"))
            except ValueError:
                learner.log_prediction({
                    "timestamp": ts,
                    "direction": row.get("direction", "FLAT"),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "magnitude_pct": float(row.get("magnitude_pct", 0.0) or 0.0),
                    "regime": row.get("regime", "Unknown"),
                    "sentiment": {"score": float(row.get("sentiment_score", 0.0) or 0.0), "n_articles": 0},
                    "macro": {"india_vix": row.get("india_vix")},
                })
                learner.update_outcome(ts, actual_return=actual, realized_pnl=pnl_val, regime=row.get("regime", "Unknown"))
            updated += 1

    print(f"Updated feedback state with {updated} matched rows.")
    print(learner.recommendations())


if __name__ == "__main__":
    main()
