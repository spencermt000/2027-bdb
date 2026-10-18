"""Baseline experiment: LightGBM quantile models on the conditioning vector.

Needs build_conditioning.py first. Writes outputs/baseline/attempt_scores.parquet and report.md.
Usage: python src/process/run_baseline.py
"""
import numpy as np
import pandas as pd

from baseline import feature_columns, oof_predictions, pinball, score_attempts
from config import paths, pipeline_cfg, seed_everything
from runlog import Timer, write_report
from schema import write_scores


def main() -> None:
    cfg = pipeline_cfg()
    seed = cfg["seed"]
    seed_everything(seed)
    p = paths()
    cond = pd.read_parquet(p["derived"] / "conditioning.parquet")
    folds = pd.read_parquet(p["derived"] / "folds.parquet")
    data = cond[cond["eval_ok"] & cond["t_total"].notna()].reset_index(drop=True)
    numeric, binary = feature_columns(cond)
    levels = cfg["eval"]["quantiles"]

    with Timer() as tm:
        q = oof_predictions(data, folds, numeric, binary, cfg["eval"], seed, "kfold")
        q_lodo = oof_predictions(data, folds, numeric, binary, cfg["eval"], seed, "lodo")
    scores = score_attempts(data, q, q_lodo)
    path = write_scores(scores, "baseline")

    y = data["t_total"].to_numpy()
    body = ["## Held-out quantile accuracy", "", "| split | MAE of q50 (s) | pinball q10 / q50 / q90 | 10-90 coverage (nominal 0.80) |", "|---|---|---|---|"]
    for name, qq in (("GroupKFold(5) by nfl_id", q), ("leave-one-draft-year-out", q_lodo)):
        v = qq.to_numpy()
        pb = pinball(y, v, levels)
        body.append(f"| {name} | {np.mean(np.abs(v[:, 1] - y)):.3f} | {pb[0]:.3f} / {pb[1]:.3f} / {pb[2]:.3f} | {np.mean((y >= v[:, 0]) & (y <= v[:, 2])):.2f} |")
    body += ["", f"Reference: SD of `t_total` = {y.std():.3f} s; MAE of predicting the training-free global median = {np.mean(np.abs(y - np.median(y))):.3f} s (in-sample, optimistic).",
             f"Features: {len(numeric)} numeric + {len(binary)} binary. Attempts scored: {len(data)}."]
    write_report("baseline", cfg["eval"], seed, len(data), data["nfl_id"].nunique(), tm.seconds, "\n".join(body))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
