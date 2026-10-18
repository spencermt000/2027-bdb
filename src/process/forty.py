"""Stage A: forty-yard dash sprint fit, QC and player-level reference (spec 1.0).

Model: v(t) = v_max * (1 - exp(-(t - t0)/tau)), t >= t0.
Distance: x(t) = v_max * (u - tau * (1 - exp(-u/tau))), u = t - t0.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from axis import project
from units import LB_TO_KG, YD_TO_M

PARAMS = ["v_max", "tau", "a0", "F0", "P_max", "t90", "d90", "speed_drop", "t10_model", "t40_model"]
FIT_BOUNDS = ([5.0, 0.2, -2.0], [16.0, 5.0, 1.5])  # v_max, tau, t0


def x_ref(u, v_max, tau):
    """Distance covered u seconds after t0 under the sprint model (yd)."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, None)
    return v_max * (u - tau * (1.0 - np.exp(-u / tau)))


def t_ref(d, v_max, tau):
    """Numerical inverse of x_ref: seconds after t0 to cover d yards. Vectorized bisection."""
    d = np.asarray(d, dtype=float)
    lo, hi = np.zeros_like(d), np.full_like(d, 1.0)
    while np.any(x_ref(hi, v_max, tau) < d):
        hi = np.where(x_ref(hi, v_max, tau) < d, hi * 2.0, hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = x_ref(mid, v_max, tau) < d
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def detect_start(t, s, thresh: float, min_frames: int):
    """Time of the first frame starting a run of >= min_frames frames with s > thresh (or None)."""
    above = s > thresh
    for i in range(len(s) - min_frames + 1):
        if above[i:i + min_frames].all():
            return float(t[i])
    return None


def fit_sprint(t, s):
    """Nonlinear least squares for (v_max, tau, t0) on speed; two starting t0 guesses."""
    def resid(th):
        return th[0] * (1.0 - np.exp(-np.clip(t - th[2], 0.0, None) / th[1])) - s

    best = None
    for t0g in (-0.3, 0.0):
        sol = least_squares(resid, [s.max() + 1.0, 1.5, t0g], bounds=FIT_BOUNDS)
        if best is None or sol.cost < best.cost:
            best = sol
    rmse = float(np.sqrt(2.0 * best.cost / len(t)))
    return best.x, rmse


def _crossing_time(t, p, level):
    """Linearly interpolated time at which p first reaches level (NaN if never)."""
    idx = np.flatnonzero(p >= level)
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (level - p[i - 1]) / (p[i] - p[i - 1]) * (t[i] - t[i - 1]))


def analyze_attempt(frames: pd.DataFrame, mass_kg: float, cfg: dict) -> dict:
    """Fit one forty attempt and derive its metrics. `cfg` is pipeline.yaml['forty']."""
    t = (frames["time"] - frames["time"].iloc[0]).dt.total_seconds().to_numpy()
    s = frames["s"].to_numpy(dtype=float)
    p, _ = project(frames[["x", "y"]].to_numpy(dtype=float))

    (v_max, tau, t0), rmse = fit_sprint(t, s)
    p_start = float(np.mean(p - x_ref(t - t0, v_max, tau)))  # start line implied by the fit
    p_rel = p - p_start
    t_start_thresh = detect_start(t, s, cfg["start_speed_thresh"], cfg["start_min_frames"])

    L = cfg["dash_length"]
    t_cross = _crossing_time(t, p_rel, L)
    t10_obs = _crossing_time(t, p_rel, 10.0) - t0
    t40_obs = t_cross - t0
    s_end = float(np.interp(t_cross, t, s)) if np.isfinite(t_cross) else float(s[-1])
    s_peak = float(s.max())

    a0 = v_max / tau
    t90 = -tau * np.log(0.1)
    out = {
        "n_frames": len(t), "p_span": float(p_rel[-1]), "p_start": p_start, "reached_40": bool(np.isfinite(t_cross)),
        "t_start_thresh": t_start_thresh, "starts_in_motion": bool(t_start_thresh == 0.0),
        "v_max": v_max, "tau": tau, "t0": t0, "a0": a0, "fit_rmse": rmse,
        "t90": t90, "d90": float(x_ref(t90, v_max, tau)),
        "speed_drop": (s_peak - s_end) / s_peak,
        "t10_model": float(t_ref(10.0, v_max, tau)), "t40_model": float(t_ref(L, v_max, tau)),
        "t10_obs": t10_obs, "t40_obs": t40_obs,
        "mass_kg": mass_kg,
    }
    out["F0"] = mass_kg * a0 * YD_TO_M
    out["P_max"] = out["F0"] * v_max * YD_TO_M / 4.0
    return out


def run_qc(att: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add residuals vs official times and qc_pass / qc_reason.

    Model times run a near-constant offset above the official ones (the official clock starts
    after motion onset), so residuals are centered on the cohort median before thresholding.
    """
    att = att.copy()
    att["resid_t10"] = att["t10_model"] - att["ten_yd_split"]
    att["resid_t40"] = att["t40_model"] - att["forty"]
    for k in ("t10", "t40"):
        att[f"resid_{k}_c"] = att[f"resid_{k}"] - att[f"resid_{k}"].median()

    lo, hi = FIT_BOUNDS
    at_bound = pd.Series(False, index=att.index)
    for col, lo_b, hi_b in zip(["v_max", "tau", "t0"], lo, hi):
        at_bound |= (att[col] <= lo_b + 1e-3) | (att[col] >= hi_b - 1e-3)
    flags = {
        "high_rmse": att["fit_rmse"] > cfg["max_fit_rmse"],
        "param_at_bound": at_bound,
        "time_resid": (att["resid_t40_c"].abs() > cfg["max_time_resid"])
        | (att["resid_t10_c"].abs() > cfg["max_time_resid"]),
        "short_run": ~att["reached_40"] & (att["p_span"] < cfg["dash_length"] - 1.5),
    }
    reasons = pd.DataFrame(flags)
    att["qc_reason"] = reasons.apply(lambda r: ",".join(r.index[r]), axis=1)
    att["qc_pass"] = att["qc_reason"] == ""
    return att


def player_reference(att: pd.DataFrame) -> pd.DataFrame:
    """Per-player forty reference: mean over QC-passing attempts, plus best attempt (fastest t40_model)."""
    ok = att[att["qc_pass"]]
    mean = ok.groupby("nfl_id")[PARAMS].mean()
    best = ok.loc[ok.groupby("nfl_id")["t40_model"].idxmin()].set_index("nfl_id")[PARAMS].add_prefix("best_")
    counts = pd.DataFrame({"n_attempts": att.groupby("nfl_id").size(), "n_pass": ok.groupby("nfl_id").size()})
    return counts.join(mean).join(best).reset_index()


def build_forty(tracking: pd.DataFrame, results: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage A end to end. `tracking` holds FORTY_YARD_DASH player frames sorted by (event_id, time)."""
    res = results.set_index("nfl_id")
    rows = []
    for event_id, f in tracking.groupby("event_id", sort=False):
        nfl_id = int(f["nfl_id"].iloc[0])
        w = res["combine_weight"].get(nfl_id, np.nan)
        row = analyze_attempt(f, w * LB_TO_KG, cfg)
        row.update(nfl_id=nfl_id, event_id=event_id, attempt=int(f["attempt"].iloc[0]),
                   draft_year=int(f["draft_year"].iloc[0]))
        rows.append(row)
    att = pd.DataFrame(rows).join(res[["ten_yd_split", "forty"]], on="nfl_id")
    att = run_qc(att, cfg)
    lead = ["nfl_id", "event_id", "attempt", "draft_year"]
    att = att[lead + [c for c in att.columns if c not in lead]]
    return att, player_reference(att)
