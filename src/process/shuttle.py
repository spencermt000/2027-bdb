"""Stage B: short shuttle (5-10-5) smoothing, segmentation, attempt features and frame table (spec 1.0).

Conventions
- Position p is along the principal axis, oriented so +p points toward +x on the field.
- Plants are the two turning points of p. Legs: leg1 = start -> plant 1, leg2 = plant 1 -> plant 2,
  leg3 = plant 2 -> finish (crossing the start line).
- Phases (frame table `phase_id`): 0 leg1, 1 plant1, 2 leg2, 3 plant2, 4 leg3. A plant phase runs from braking onset
  to `reaccel_window` seconds after the turn (or the end of the dwell, if later). `subphase_id`: 0 free, 1 brake,
  2 dwell, 3 reaccel.
- `*_rel` columns are sign-normalized so the first leg heads in the +direction (comparable across L/R starts).
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from axis import principal_axis
from forty import t_ref
from smoothing import smooth_xy
from units import LB_TO_KG, YD_TO_M

QC_COLS = ["qc_pass", "qc_reason"]


def oriented_axis(xy: np.ndarray) -> np.ndarray:
    """Principal axis with a fixed sign convention (+x on the field)."""
    ax = principal_axis(xy)
    return ax if ax[0] >= 0 else -ax


def find_turns(t, p, v, prominence: float):
    """Turning points of p as (times, positions, indices, n_found). Keeps the two most prominent, in time order."""
    cands = []
    for sgn in (1, -1):
        idx, props = find_peaks(sgn * p, prominence=prominence)
        cands += [(i, pr) for i, pr in zip(idx, props["prominences"])]
    n_found = len(cands)
    cands = sorted(sorted(cands, key=lambda c: -c[1])[:2])
    times, pos, idxs = [], [], []
    for i, _ in cands:
        # refine to the sign change of v nearest the extremum (sub-frame), else keep the extremum frame
        tt = float(t[i])
        for j in sorted(range(max(0, i - 3), min(len(t) - 1, i + 3)), key=lambda j: abs(j - i)):
            if v[j] * v[j + 1] < 0:
                tt = float(t[j] + v[j] / (v[j] - v[j + 1]) * (t[j + 1] - t[j]))
                break
        times.append(tt)
        pos.append(float(np.interp(tt, t, p)))
        idxs.append(int(i))
    return times, pos, idxs, n_found


def _cross(t, y, level, direction, start_idx=0):
    """Interpolated time of the first crossing of `level` moving in `direction` (+1/-1) at or after start_idx."""
    z = (y - level) * direction
    for i in range(max(start_idx, 1), len(t)):
        if z[i - 1] < 0 <= z[i]:
            return float(t[i - 1] + (-z[i - 1]) / (z[i] - z[i - 1]) * (t[i] - t[i - 1]))
    return np.nan


def _plant_features(k, t, p, v, a, t_from, p_from, t_turn, p_turn, dir_out, mass_kg, a0, vref_fn, cfg):
    """Braking, dwell and re-acceleration features for the plant at t_turn. dir_out is the exit direction (+1/-1)."""
    d_in = -dir_out
    vd, ad = v * d_in, a * d_in                      # along the direction of entry
    seg = np.flatnonzero((t >= t_from) & (t <= t_turn))
    if len(seg) < 2:
        return {}, None
    i_pk = seg[np.argmax(vd[seg])]
    entry_speed = float(vd[i_pk])
    j = seg[-1]                                      # last frame at or before the turn
    onset = j
    while onset - 1 >= i_pk and ad[onset] < -cfg["brake_decel_thresh"]:
        onset -= 1
    if ad[onset] >= -cfg["brake_decel_thresh"] and onset < j:
        onset += 1
    t_on = float(t[onset])

    # dwell: contiguous interval around the turn with |v| < dwell_speed (interpolated edges)
    thr = cfg["dwell_speed"]
    i_t = int(np.argmin(np.abs(t - t_turn)))
    dwell, t_ds, t_de = 0.0, t_turn, t_turn
    if abs(v[i_t]) < thr:
        lo, hi = i_t, i_t
        while lo - 1 >= 0 and abs(v[lo - 1]) < thr:
            lo -= 1
        while hi + 1 < len(t) and abs(v[hi + 1]) < thr:
            hi += 1
        t_ds = float(t[lo]) if lo == 0 else float(np.interp(thr, [abs(v[lo - 1]), abs(v[lo])][::-1], [t[lo - 1], t[lo]][::-1]))
        t_de = float(t[hi]) if hi == len(t) - 1 else float(np.interp(thr, [abs(v[hi + 1]), abs(v[hi])], [t[hi + 1], t[hi]]))
        dwell = max(t_de - t_ds, 0.0)
    t_be = min(max(t_ds, t_on), t_turn)               # braking ends where the dwell begins
    brake_time = t_be - t_on
    in_brake = (t >= t_on) & (t <= t_be)
    v_on, v_be = float(np.interp(t_on, t, vd)), float(np.interp(t_be, t, vd))
    brake_ok = brake_time > 1e-6 and in_brake.any()

    W = cfg["reaccel_window"]
    grid = np.linspace(t_turn, t_turn + W, 21)
    reaccel = float(np.mean(np.interp(grid, t, a * dir_out)))

    feats = {
        f"entry_speed_{k}": entry_speed,
        f"brake_onset_dist_{k}": abs(p_turn - float(np.interp(t_on, t, p))),
        f"brake_dist_{k}": abs(float(np.interp(t_be, t, p)) - float(np.interp(t_on, t, p))),
        f"brake_time_{k}": brake_time,
        f"mean_decel_{k}": (v_on - v_be) / brake_time if brake_ok else np.nan,
        f"peak_decel_{k}": float(np.max(-ad[in_brake])) if brake_ok else np.nan,
        f"brake_impulse_{k}": mass_kg * (v_on - v_be) * YD_TO_M,
        f"dwell_{k}": dwell,
        f"reaccel_{k}": reaccel,
        f"reaccel_ratio_{k}": reaccel / a0 if np.isfinite(a0) else np.nan,
        f"t_brake_onset_{k}": t_on,
        f"leg_dist_{k}": abs(p_turn - p_from),
    }
    d_leg = feats[f"leg_dist_{k}"] if cfg["entry_ratio_dist"] == "observed" else cfg["leg_dist"][k - 1]
    feats[f"entry_ratio_{k}"] = entry_speed / vref_fn(d_leg) if vref_fn is not None else np.nan
    win_end = max(t_turn + W, t_de)
    return feats, dict(t_on=t_on, t_be=t_be, t_de=t_de, t_turn=t_turn, win_end=win_end)


def analyze_attempt(frames: pd.DataFrame, mass_kg: float, ref: dict | None, cfg: dict):
    """Smooth, segment and featurize one shuttle attempt.

    ref: player-level forty parameters (v_max, tau, a0) or None. Returns (feature row dict, frame DataFrame or None).
    """
    t = (frames["time"] - frames["time"].iloc[0]).dt.total_seconds().to_numpy()
    xy = frames[["x", "y"]].to_numpy(dtype=float)
    row = {"n_frames": len(t), "t_total": np.nan}
    reasons = []
    if len(t) < cfg["min_frames"]:
        row.update(qc_pass=False, qc_plants_ok=False, qc_reason="too_few_frames")
        return row, None

    ax = oriented_axis(xy)
    sm = smooth_xy(xy, t, ax, cfg["smoother_sigma"], cfg["smoother_q"])
    p, v, a = sm["p"], sm["v"], sm["a"]
    times, pos, _, n_found = find_turns(t, p, v, cfg["min_turn_prominence"])
    row["n_plants_found"] = n_found
    if n_found != 2:
        reasons.append(f"n_plants={n_found}")
    if len(times) < 2:
        row.update(qc_pass=False, qc_plants_ok=False, qc_reason=",".join(reasons))
        return row, None
    (t1, t2), (p1, p2) = times, pos

    # start: first sustained rise in smoothed speed, then back-extrapolate speed along the first leg to zero
    dir1 = float(np.sign(p1 - p[0])) or 1.0
    above = sm["speed"] > cfg["start_speed_thresh"]
    k = cfg["start_min_frames"]
    i0 = next((i for i in range(len(t) - k + 1) if above[i:i + k].all()), 0)
    row["start_idx"] = i0
    m = cfg["start_fit_frames"]
    slope, icpt = np.polyfit(t[i0:i0 + m] - t[i0], (v * dir1)[i0:i0 + m], 1)
    back = float(np.clip(icpt / slope, 0.0, cfg["max_start_backoff"])) if slope > 0 and icpt > 0 else 0.0
    t_start = float(t[i0]) - back
    p_line = float(p[i0] - dir1 * (icpt**2 / (2 * slope) if back > 0 else 0.0))
    dir2, dir3 = -dir1, dir1
    t_end = _cross(t, p, p_line, dir3, start_idx=int(np.searchsorted(t, t2)))
    if not np.isfinite(t_end):
        reasons.append("no_finish")
    t_total = t_end - t_start
    finish_speed = abs(float(np.interp(t_end, t, v))) if np.isfinite(t_end) else np.nan
    if np.isfinite(finish_speed) and finish_speed < cfg["min_finish_speed"]:
        reasons.append("slow_finish")

    a0 = ref["a0"] if ref else np.nan
    vref_fn = (lambda d: ref["v_max"] * (1.0 - np.exp(-float(t_ref(d, ref["v_max"], ref["tau"])) / ref["tau"]))) if ref else None
    f1, w1 = _plant_features(1, t, p, v, a, t_start, p_line, t1, p1, dir2, mass_kg, a0, vref_fn, cfg)
    f2, w2 = _plant_features(2, t, p, v, a, t1, p1, t2, p2, dir3, mass_kg, a0, vref_fn, cfg)
    if not f1 or not f2:
        row.update(qc_pass=False, qc_plants_ok=False, qc_reason=",".join(reasons + ["plant_features"]))
        return row, None
    row["qc_plants_ok"] = n_found == 2 and bool(t_start < t1 < t2)

    row.update({"t_total": t_total, "finish_speed": finish_speed, "t_start": t_start, "t_end": t_end, "t_plant_1": t1, "t_plant_2": t2,
                "t_leg1": t1 - t_start, "t_leg2": t2 - t1, "t_leg3": t_end - t2,
                "start_dir": "R" if dir1 > 0 else "L", "p_line": p_line, "p_plant_1": p1, "p_plant_2": p2})
    row.update(f1)
    row.update(f2)
    row["leg_dist_3"] = abs(p_line - p2)
    if ref:
        legs = np.array([row["t_leg1"], row["t_leg2"], row["t_leg3"]])
        t_nom = np.array([float(t_ref(d, ref["v_max"], ref["tau"])) for d in cfg["leg_dist"]])
        excess = legs - t_nom
        row["cod_cost"] = float(excess.sum())
        row["cod_cost_leg_1"], row["cod_cost_leg_2"], row["cod_cost_leg_3"] = (float(e) for e in excess)
        row["cod_cost_plant_1"] = float(excess[0] + excess[1] / 2)
        row["cod_cost_plant_2"] = float(excess[1] / 2 + excess[2])
        row["t_ref_total"] = float(t_nom.sum())
    else:
        reasons.append("no_forty_ref")

    if np.isfinite(t_total) and not (cfg["min_t_total"] <= t_total <= cfg["max_t_total"]):
        reasons.append("t_total_range")
    if not (t_start < t1 < t2):
        reasons.append("plant_order")
    row.update(qc_reason=",".join(reasons), qc_pass=len(reasons) == 0)

    # frame table: frames from the start to the finish crossing
    keep = (t >= t_start) & (t <= (t_end if np.isfinite(t_end) else t[-1]))
    tk = t[keep]
    nxt = np.where(tk < t1, p1, np.where(tk < t2, p2, p_line))
    dirs = np.where(tk < t1, dir1, np.where(tk < t2, dir2, dir3))
    phase = np.zeros(len(tk), dtype=int)
    sub = np.zeros(len(tk), dtype=int)
    for pid, w in ((1, w1), (3, w2)):
        in_plant = (tk >= w["t_on"]) & (tk < w["win_end"])
        phase[in_plant] = pid
        sub[in_plant & (tk >= w["t_on"])] = 1
        sub[in_plant & (tk >= w["t_be"])] = 2
        sub[in_plant & (tk >= w["t_de"])] = 3
    phase[(tk >= w1["win_end"]) & (tk < w2["t_on"])] = 2
    phase[tk >= w2["win_end"]] = 4
    phase[(tk >= w1["win_end"]) & (tk < w2["t_on"])] = 2
    fr = pd.DataFrame({
        "frame_idx": np.arange(keep.sum()), "t": tk, "t_since_start": tk - t_start,
        "p": p[keep], "v": v[keep], "a": a[keep],
        "p_var": sm["p_var"][keep], "v_var": sm["v_var"][keep], "a_var": sm["a_var"][keep],
        "speed": sm["speed"][keep], "d_line": (nxt - p[keep]) * dirs,
        "phase_id": phase, "subphase_id": sub,
        "p_rel": (p[keep] - p_line) * dir1, "v_rel": v[keep] * dir1, "a_rel": a[keep] * dir1,
    })
    return row, fr


def build_shuttle(tracking: pd.DataFrame, results: pd.DataFrame, forty_ref: pd.DataFrame, cfg: dict):
    """Stage B end to end. Returns (attempt feature table, frame table).

    forty_ref: forty_reference.parquet contents; `cfg['forty_reference']` selects mean or best columns.
    """
    res = results.set_index("nfl_id")
    prefix = "best_" if cfg["forty_reference"] == "best" else ""
    refs = {}
    for r in forty_ref[forty_ref["n_pass"] > 0].itertuples(index=False):
        refs[r.nfl_id] = {k: getattr(r, prefix + k) for k in ("v_max", "tau", "a0")}
    rows, frames = [], []
    for event_id, f in tracking.groupby("event_id", sort=False):
        nfl_id = int(f["nfl_id"].iloc[0])
        w = res["combine_weight"].get(nfl_id, np.nan)
        row, fr = analyze_attempt(f, w * LB_TO_KG, refs.get(nfl_id), cfg)
        row.update(nfl_id=nfl_id, event_id=event_id, attempt=int(f["attempt"].iloc[0]),
                   draft_year=int(f["draft_year"].iloc[0]), t_first=f["time"].iloc[0])
        rows.append(row)
        if fr is not None:
            fr.insert(0, "event_id", event_id)
            fr.insert(0, "nfl_id", nfl_id)
            fr.insert(2, "attempt", row["attempt"])
            frames.append(fr)
    att = pd.DataFrame(rows).join(res[["short_shuttle"]], on="nfl_id")
    att["attempt_order"] = att.groupby("nfl_id")["t_first"].rank(method="first").astype(int)
    att["resid_t_total"] = att["t_total"] - att["short_shuttle"]
    att["resid_t_total_c"] = att["resid_t_total"] - att["resid_t_total"].median()
    bad = att["resid_t_total_c"].abs() > cfg["max_time_resid"]
    att.loc[bad, "qc_reason"] = (att.loc[bad, "qc_reason"].str.cat(pd.Series("time_resid", index=att.index[bad]), sep=",")
                                 .str.lstrip(","))
    att["qc_pass"] = att["qc_reason"] == ""
    lead = ["nfl_id", "event_id", "attempt", "attempt_order", "draft_year"]
    att = att[lead + [c for c in att.columns if c not in lead]].drop(columns="t_first")
    return att, (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
