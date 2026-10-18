"""Baseline: LightGBM quantile models (q = 0.1, 0.5, 0.9) predicting t_total from the conditioning vector.

Score = the attempt's position in the predicted distribution. Out-of-fold predictions use the shared grouped folds
(secondary: leave-one-draft-year-out). The standardizer is refit inside each training split.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from bio import KEYS, Conditioner, assert_no_leakage, feature_columns


def quantile_position(y: np.ndarray, q: np.ndarray, levels=(0.1, 0.5, 0.9), floor=0.02, clip=(0.01, 0.99)) -> np.ndarray:
    """Piecewise-linear CDF position of y through predicted quantiles q (n, 3), extended linearly past the outer knots."""
    q = np.sort(q, axis=1)
    lo_w = np.maximum(q[:, 1] - q[:, 0], floor)
    hi_w = np.maximum(q[:, 2] - q[:, 1], floor)
    below = levels[1] + (y - q[:, 1]) * (levels[1] - levels[0]) / lo_w
    above = levels[1] + (y - q[:, 1]) * (levels[2] - levels[1]) / hi_w
    return np.clip(np.where(y < q[:, 1], below, above), *clip)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, numeric, binary, cfg: dict, seed: int) -> np.ndarray:
    """Fit three quantile models on `train` (needs t_total) and return (len(test), 3) predictions."""
    cond = Conditioner(numeric, binary).fit(train)
    Xtr, Xte = cond.transform(train), cond.transform(test)
    assert_no_leakage(Xtr.columns)
    preds = []
    for a in cfg["quantiles"]:
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=seed, verbose=-1, **cfg["lgbm"])
        preds.append(m.fit(Xtr, train["t_total"]).predict(Xte))
    return np.sort(np.column_stack(preds), axis=1)


def oof_predictions(data: pd.DataFrame, folds: pd.DataFrame, numeric, binary, cfg: dict, seed: int, mode: str = "kfold") -> pd.DataFrame:
    """Out-of-fold quantile predictions for every row of `data` (QC-passing attempts with a fold/draft_year)."""
    group = "fold" if mode == "kfold" else "draft_year"
    data = data.merge(folds[["nfl_id", "fold", "draft_year"]].drop_duplicates("nfl_id"), on="nfl_id", how="left", suffixes=("", "_f"))
    out = np.full((len(data), len(cfg["quantiles"])), np.nan)
    for g in sorted(data[group].unique()):
        te = (data[group] == g).to_numpy()
        out[te] = fit_predict(data[~te], data[te], numeric, binary, cfg, seed)
    return pd.DataFrame(out, index=data.index, columns=[f"q{int(a * 100)}" for a in cfg["quantiles"]])


def score_attempts(data: pd.DataFrame, q: pd.DataFrame, q_lodo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Standard output format for exp_id = 'baseline'. score_primary > 0: faster than the predicted median region."""
    y = data["t_total"].to_numpy()
    qv = q.to_numpy()
    u = quantile_position(y, qv)
    out = data[KEYS].reset_index(drop=True).copy()
    out["exp_id"] = "baseline"
    out["score_primary"] = 0.5 - u
    out["score_quantile_pos"] = u
    out["score_resid_median"] = qv[:, 1] - y        # seconds faster than predicted median
    out["score_pred_q10"], out["score_pred_q50"], out["score_pred_q90"] = qv[:, 0], qv[:, 1], qv[:, 2]
    if q_lodo is not None:
        out["score_lodo"] = 0.5 - quantile_position(y, q_lodo.to_numpy())
    for c in out.columns:
        if c.startswith("score_"):
            out[c] = out[c].astype(float)
    return out


def pinball(y: np.ndarray, q: np.ndarray, levels) -> np.ndarray:
    """Mean pinball loss per quantile level."""
    return np.array([np.mean(np.maximum(a * (y - q[:, i]), (a - 1) * (y - q[:, i]))) for i, a in enumerate(levels)])


__all__ = ["feature_columns", "fit_predict", "oof_predictions", "score_attempts", "quantile_position", "pinball"]
