"""Evaluation harness: reliability, incremental validity, held-out fit, cross-experiment agreement, leakage check.

Reads every outputs/<exp_id>/attempt_scores.parquet present, so it works with any subset of experiments.
Convention for held-out fit: an experiment may add `score_ll_heldout` (per-attempt predictive log-likelihood on
held-out players); the report averages it where present.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from schema import EXP_IDS, KEYS, validate_scores


def icc1(scores: pd.Series, groups: pd.Series) -> float:
    """One-way random-effects ICC(1) for unbalanced groups, using only groups with 2+ observations."""
    df = pd.DataFrame({"y": scores.to_numpy(), "g": groups.to_numpy()})
    df = df[df.groupby("g")["y"].transform("size") >= 2]
    k, N = df["g"].nunique(), len(df)
    if k < 2 or N <= k:
        return np.nan
    gm = df.groupby("g")["y"].agg(["mean", "size"])
    grand = df["y"].mean()
    msb = (gm["size"] * (gm["mean"] - grand) ** 2).sum() / (k - 1)
    msw = ((df["y"] - df["g"].map(gm["mean"])) ** 2).sum() / (N - k)
    n0 = (N - (gm["size"] ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else np.nan


def _first_two(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Players with 2+ attempts: attempt_order 1 and 2 scores side by side."""
    w = df.sort_values("attempt_order").groupby("nfl_id").head(2).copy()
    w["k"] = w.groupby("nfl_id").cumcount()
    w = w.pivot(index="nfl_id", columns="k", values=col)
    return w.dropna() if w.shape[1] == 2 else w.iloc[:0]


def _boot(fn, players: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(rng.choice(players, size=len(players), replace=True))
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 20 else (np.nan, np.nan)


def reliability(df: pd.DataFrame, col: str = "score_primary", n_boot: int = 1000, seed: int = 0) -> dict:
    """ICC(1) over players with 2+ attempts and Spearman of attempt-1 vs attempt-2 scores, with player-bootstrap CIs."""
    multi = df[df.groupby("nfl_id")["event_id"].transform("size") >= 2]
    players = multi["nfl_id"].unique()
    out = {"n_players": len(players), "icc": np.nan, "icc_ci": (np.nan, np.nan), "spearman_12": np.nan, "spearman_ci": (np.nan, np.nan)}
    if len(players) < 3:
        return out
    out["icc"] = icc1(multi[col], multi["nfl_id"])
    pair = _first_two(multi, col)
    out["spearman_12"] = float(spearmanr(pair[0], pair[1])[0]) if len(pair) >= 3 else np.nan
    by = {p: g for p, g in multi.groupby("nfl_id")}

    def boot_icc(ps):
        parts = [by[p].assign(_b=i) for i, p in enumerate(ps)]
        d = pd.concat(parts)
        return icc1(d[col], d["_b"])

    def boot_sp(ps):
        d = pair.loc[[p for p in ps if p in pair.index]]
        return float(spearmanr(d[0], d[1])[0]) if len(d) >= 3 else np.nan

    out["icc_ci"] = _boot(boot_icc, players, n_boot, seed)
    if len(pair) >= 3:
        out["spearman_ci"] = _boot(boot_sp, pair.index.to_numpy(), n_boot, seed)
    return out


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]


def incremental_validity(player: pd.DataFrame, outcome: str, controls: list[str], col: str = "score") -> dict:
    """Partial correlation of the player-level score with `outcome` given `controls`, and R^2 gain of adding the score."""
    d = player[[col, outcome, *controls]].dropna()
    n = len(d)
    out = {"outcome": outcome, "n": n, "partial_r": np.nan, "r2_gain": np.nan, "partial_ci": (np.nan, np.nan)}
    if n < len(controls) + 5:
        return out
    X, s, y = d[controls].to_numpy(float), d[col].to_numpy(float), d[outcome].to_numpy(float)

    def pr(idx):
        a, b = _residualize(s[idx], X[idx]), _residualize(y[idx], X[idx])
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan

    def r2(Z, idx):
        e = _residualize(y[idx], Z[idx])
        return 1.0 - (e @ e) / (((y[idx] - y[idx].mean()) ** 2).sum())

    idx = np.arange(n)
    out["partial_r"] = pr(idx)
    out["r2_gain"] = r2(np.column_stack([X, s]), idx) - r2(X, idx)
    rng = np.random.default_rng(0)
    bs = [pr(rng.integers(0, n, n)) for _ in range(500)]
    bs = [b for b in bs if np.isfinite(b)]
    if len(bs) > 20:
        out["partial_ci"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


def player_scores(df: pd.DataFrame, col: str = "score_primary") -> pd.Series:
    return df.groupby("nfl_id")[col].mean()


def agreement(per_exp: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman matrix of player-level scores (pairwise complete) and the players with the largest rank disagreement."""
    wide = pd.DataFrame(per_exp)
    mat = wide.corr(method="spearman", min_periods=5)
    pct = wide.rank(pct=True)
    complete = pct.dropna()
    spread = (complete.max(axis=1) - complete.min(axis=1)).sort_values(ascending=False) if wide.shape[1] > 1 else pd.Series(dtype=float)
    top = pd.concat([complete.loc[spread.index[:10]].round(2), spread.head(10).rename("rank_range").round(2)], axis=1) if len(spread) else pd.DataFrame()
    return mat, top


def leakage_check(per_exp: dict[str, pd.Series], ngs: pd.Series) -> pd.DataFrame:
    rows = []
    for e, s in per_exp.items():
        d = pd.concat([s.rename("s"), ngs.rename("ngs")], axis=1).dropna()
        rows.append({"exp_id": e, "n": len(d), "spearman": spearmanr(d["s"], d["ngs"])[0] if len(d) > 3 else np.nan,
                     "pearson": d["s"].corr(d["ngs"]) if len(d) > 3 else np.nan})
    return pd.DataFrame(rows).set_index("exp_id")


def load_all_scores(outputs_dir) -> dict[str, pd.DataFrame]:
    found = {}
    for e in sorted(EXP_IDS):
        f = outputs_dir / e / "attempt_scores.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            validate_scores(df, e)
            found[e] = df
    return found


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def build_report(scores: dict[str, pd.DataFrame], context: pd.DataFrame, results: pd.DataFrame, cfg: dict, seed: int) -> str:
    """Comparison report (markdown). `context` has KEYS + attempt_order + t_total + f40_* per attempt."""
    L = ["# Comparison report", "", f"Experiments present: {', '.join(scores) or 'none'}", ""]
    if not scores:
        return "\n".join(L + ["No `attempt_scores.parquet` files found."])
    res = results.set_index("nfl_id")
    ctx = context.set_index(KEYS)
    joined = {}
    for e, df in scores.items():
        j = df.join(ctx[["attempt_order", "t_total"]], on=KEYS)
        joined[e] = j
    L += ["## Sample", ""] + [f"- {e}: {len(d)} attempts, {d['nfl_id'].nunique()} players" for e, d in joined.items()] + [""]

    L += ["## Reliability (players with 2+ attempts)", "",
          "| exp | players | ICC(1) [95% CI] | Spearman att1 vs att2 [95% CI] |", "|---|---|---|---|"]
    for e, d in joined.items():
        r = reliability(d, n_boot=cfg["n_boot"], seed=seed)
        L.append(f"| {e} | {r['n_players']} | {_fmt(r['icc'])} [{_fmt(r['icc_ci'][0])}, {_fmt(r['icc_ci'][1])}] | "
                 f"{_fmt(r['spearman_12'])} [{_fmt(r['spearman_ci'][0])}, {_fmt(r['spearman_ci'][1])}] |")
    L += ["", "Few players have 2+ QC-passing attempts, so the intervals are wide.", ""]

    per_exp = {e: player_scores(d) for e, d in joined.items()}
    base_ctrl = ["t_total", "f40_v_max", "f40_tau"]
    pl = context.groupby("nfl_id").agg(t_total=("t_total", "mean"), f40_v_max=("f40_v_max", "first"), f40_tau=("f40_tau", "first"))
    L += ["## Incremental validity", "",
          f"Partial correlation / R² gain of the player-level score with an external outcome, controlling for mean `t_total` and "
          f"forty reference (`v_max`, `tau`). Outcomes (never model inputs): {cfg['validity_outcomes']}. "
          "Outcome is a time (lower = better) and scores are higher = better, so a negative partial r is the expected sign.", "",
          "| exp | outcome | n | partial r [95% CI] | R² gain |", "|---|---|---|---|---|"]
    for e, s in per_exp.items():
        for outc in cfg["validity_outcomes"]:
            d = pl.join(s.rename("score")).join(res[[outc]])
            v = incremental_validity(d, outc, base_ctrl)
            L.append(f"| {e} | {outc} | {v['n']} | {_fmt(v['partial_r'])} [{_fmt(v['partial_ci'][0])}, {_fmt(v['partial_ci'][1])}] | {_fmt(v['r2_gain'])} |")
    L.append("")

    L += ["## Held-out fit", ""]
    ll = {e: d["score_ll_heldout"].mean() for e, d in joined.items() if "score_ll_heldout" in d}
    L += [f"- {e}: mean held-out log-likelihood per attempt {_fmt(v)}" for e, v in ll.items()] or \
         ["No experiment reports `score_ll_heldout` yet (defined for T1, T2, T5). Baseline quantile accuracy is in `outputs/baseline/report.md`."]
    L.append("")

    mat, top = agreement(per_exp)
    L += ["## Cross-experiment agreement (Spearman of player-level scores)", "", "```", mat.round(3).to_string(), "```", ""]
    if len(top):
        L += ["Players with the largest disagreement (percentile ranks by experiment):", "", "```", top.to_string(), "```", ""]
    else:
        L += ["Only one experiment present; no disagreement list.", ""]

    lk = leakage_check(per_exp, res["ngs_athleticism_score"])
    L += ["## Leakage check: correlation with `ngs_athleticism_score`", "",
          "NGS athleticism likely includes shuttle results, so a high correlation is expected. Reported only.", "",
          "```", lk.round(3).to_string(), "```", ""]
    return "\n".join(L)
