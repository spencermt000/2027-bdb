'''
02_compute_metrics.py

input: comes from @01_standardize.py  (processed/game_tracking_std_{season}.parquet)
       or, with --source combine, raw/combine_tracking.csv (PLAYER rows, all drills; there is no `o`)
goal: compute the metrics to be used as features for any downstream tasks

A trajectory is one player's continuous track: (game_id, play_id, nfl_id) for game tracking, event_id for combine.
Three outputs per source (processed/):
  *_metrics_frames   one row per frame: kinematics, cumulative, rolling, forecast-error and cut-timing features
  *_metrics_traj     one row per trajectory: path geometry, speed profile, smoothness, complexity, cuts, quality flags
  *_metrics_cuts     one row per detected change of direction

Conventions
- `dir` / `o` are degrees clockwise from +y (checked against positions: median error 0.7 deg). Positive turn = clockwise.
- Distances come from positions. `dis` is NOT a Euclidean step: dis / step rises with speed (about 0.27 at 2-4 yd/s to
  about 1.07 at 10-12 yd/s), so it is kept only as a raw ratio (`dis_over_step`), never as a distance.
- Rates use the actual dt between frames (nominally 0.1 s); irregular frames are flagged in `dt_gap`.
- Yards and yd/s throughout, except `*_per_kg` energy terms, which are SI (J/kg, W/kg).

Implemented (frame level)
- step, velocity components vx/vy, speed from positions, speed residual vs `s`, `dis` / step
- signed longitudinal acceleration, jerk (signed and of provided `a`), angular velocity / acceleration of `dir`,
  lateral (centripetal) acceleration, curvature, turn radius
- orientation offset o - dir (wrapped), angular velocity of `o`, forward / side / back band and time spent in each
- cumulative: seconds, ground covered, max speed, angular distance, net turn, net displacement, straightness,
  tortuosity, turning wiggle, time / distance at which the running max speed was set
- rolling k-frame: path length, net displacement, straightness, speed mean / std, deviation from the rolling mean,
  mean |turn rate|; EWMA of speed
- energy per unit mass: kinetic energy, instantaneous power
- constant-velocity and constant-acceleration position forecast error at 1, 3, 5 frames ahead
- time since last cut, cuts so far; game only: seconds since ball_snap
Implemented (trajectory level)
- path length, net displacement, straightness, tortuosity, total vs net turning, max perpendicular deviation from the
  start-end chord, convex hull area, radius of gyration
- max speed, time / distance to max, time to 50 / 80 / 90% of max, frames within 95% of max, decay after the peak
- max accel / decel, jerk RMS, log dimensionless jerk, SPARC, dominant frequency and spectral entropy of speed,
  sample entropy of turn rate, autocorrelation of acceleration, variance of turning rate
- orientation band fractions, o-vs-dir lag (game only)
- cuts: count, per second, left / right counts and asymmetry, mean entry / min / exit speed, retention, angle,
  time in cut, deceleration rate in, re-acceleration time out, COD deficit
- game events: speed at snap and at pass_forward, distance and time from snap to pass_forward
- quality: dropped frames, stationary position jitter, speed residual, `dis` / step
- combine only (--source combine): attempt-to-attempt change and spread, size-normalized versions, Froude number,
  z-scores within drill x position x year
Not implemented (see the original brainstorm at the bottom): fractal dimension, hip-flip time, game-vs-combine
translation ratios, get-off vs `player_get_off`, Kalman innovations.

Usage: python src/process/02_compute_metrics.py [--source game|combine] [--seasons 2023 2024 2025] [--no-frames]
'''
import argparse

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, lfilter
from scipy.spatial import ConvexHull

from config import ROOT, pipeline_cfg
from units import YD_TO_M

PROCESSED = ROOT / "processed"
GAME_KEYS = ["game_id", "play_id", "nfl_id"]
COMBINE_KEYS = ["event_id"]
G = 9.81


def wrap_deg(d):
    """Wrap an angle difference to [-180, 180)."""
    return (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------------------------------------------
# frame-level features (vectorized across all trajectories; the frame table must be sorted by keys + time)
# ---------------------------------------------------------------------------------------------------------------

def _group_layout(df: pd.DataFrame, keys: list[str]):
    """Contiguous group ids, each row's index within its group, and its group's start row."""
    gid = df.groupby(keys, sort=False).ngroup().to_numpy()
    new = np.r_[True, gid[1:] != gid[:-1]]
    if (np.bincount(gid) > 0).sum() != new.sum():
        raise ValueError("frames must be sorted so each trajectory is contiguous")
    start = np.maximum.accumulate(np.where(new, np.arange(len(df)), 0))
    return gid, np.arange(len(df)) - start, start


def _gdiff(v: np.ndarray, within: np.ndarray, lag: int = 1) -> np.ndarray:
    """v[i] - v[i-lag], NaN where i-lag is in another trajectory."""
    out = np.full(len(v), np.nan)
    out[lag:] = v[lag:] - v[:-lag]
    out[within < lag] = np.nan
    return out


def _gshift(v: np.ndarray, within: np.ndarray, gid: np.ndarray, lag: int) -> np.ndarray:
    """v[i-lag] (lag < 0 looks ahead), NaN when it crosses a trajectory boundary."""
    out = np.full(len(v), np.nan)
    if lag > 0:
        out[lag:] = v[:-lag]
        out[within < lag] = np.nan
    else:
        k = -lag
        out[:-k] = v[k:]
        ok = np.zeros(len(v), dtype=bool)
        ok[:-k] = gid[k:] == gid[:-k]
        out[~ok] = np.nan
    return out


def _rolling(v: np.ndarray, within: np.ndarray, k: int, min_periods: int):
    """Trailing-window sum, count and sum of squares inside each trajectory (NaNs ignored). Returns mean, std, sum, count."""
    ok = np.isfinite(v)
    x = np.where(ok, v, 0.0)
    c1, c2, cn = np.cumsum(x), np.cumsum(x * x), np.cumsum(ok.astype(float))
    idx = np.arange(len(v))
    lo = idx - np.minimum(within, k - 1) - 1                     # index just before the window start
    prev = lambda c: np.where(lo >= 0, c[np.maximum(lo, 0)], 0.0)  # noqa: E731
    s1, s2, n = c1 - prev(c1), c2 - prev(c2), cn - prev(cn)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = np.maximum(s2 / n - mean**2, 0.0) * n / np.maximum(n - 1, 1)
    bad = n < min_periods
    return np.where(bad, np.nan, mean), np.where(bad | (n < 2), np.nan, np.sqrt(var)), np.where(bad, np.nan, s1), n


def _ewma(v: np.ndarray, gid: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty(len(v))
    edges = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        x = np.nan_to_num(v[a:b], nan=0.0)
        out[a:b] = lfilter([alpha], [1.0, -(1 - alpha)], x, zi=[(1 - alpha) * x[0]])[0]
    return out


def add_frame_features(df: pd.DataFrame, keys: list[str], cfg: dict) -> pd.DataFrame:
    """Per-frame kinematic, cumulative, rolling and forecast features. Returns a new frame with the same row order."""
    gid, within, _ = _group_layout(df, keys)
    t = df["time"].to_numpy("datetime64[ns]")
    dt = np.where(within > 0, np.r_[np.nan, np.diff(t).astype(float) / 1e9], np.nan)
    dt = np.where(dt > 0, dt, np.nan)
    x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
    s, a = df["s"].to_numpy(float), df["a"].to_numpy(float)
    d = df["dir"].to_numpy(float)
    f = pd.DataFrame(index=df.index)

    f["dt"] = dt
    f["dt_gap"] = np.where(within > 0, ~(np.abs(dt - 0.1) <= cfg["gap_tol"]), False)
    dx, dy = _gdiff(x, within), _gdiff(y, within)
    step = np.hypot(dx, dy)
    f["step"] = step
    with np.errstate(invalid="ignore", divide="ignore"):
        f["dis_over_step"] = np.where(step > cfg["min_step"], df["dis"].to_numpy(float) / step, np.nan)
        f["speed_pos"] = step / dt
    f["speed_resid"] = s - f["speed_pos"].to_numpy()
    th = np.radians(d)
    f["vx"], f["vy"] = s * np.sin(th), s * np.cos(th)

    f["a_long"] = _gdiff(s, within) / dt
    f["jerk_long"] = _gdiff(f["a_long"].to_numpy(), within) / dt
    f["jerk_a"] = _gdiff(a, within) / dt
    d_dir = wrap_deg(_gdiff(d, within))
    omega = d_dir / dt                                              # deg/s, clockwise positive
    f["d_dir"], f["omega"] = d_dir, omega
    f["alpha"] = _gdiff(omega, within) / dt
    f["a_lat"] = s * np.radians(omega)                              # yd/s^2
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(step > cfg["min_step"], np.radians(d_dir) / step, np.nan)
        f["curvature"] = kappa
        f["turn_radius"] = np.minimum(1.0 / np.abs(kappa), cfg["max_turn_radius"])

    if "o" in df and df["o"].notna().any():
        o = df["o"].to_numpy(float)
        off = wrap_deg(o - d)
        f["o_offset"] = off
        f["omega_o"] = wrap_deg(_gdiff(o, within)) / dt
        band = np.where(np.abs(off) < cfg["band_forward_deg"], 0, np.where(np.abs(off) > cfg["band_back_deg"], 2, 1))
        f["o_band"] = np.where(np.isfinite(off), band, -1).astype(np.int8)       # 0 forward, 1 side, 2 backpedal
        for k, name in enumerate(["forward", "side", "back"]):
            hit = np.where(np.isfinite(off) & (band == k), np.nan_to_num(dt, nan=0.1), 0.0)
            f[f"cum_t_{name}"] = _group_cumsum(hit, gid)

    v_mps = s * YD_TO_M
    f["ke_per_kg"] = 0.5 * v_mps**2
    f["power_per_kg"] = f["a_long"].to_numpy() * YD_TO_M * v_mps

    # cumulative
    t0 = pd.Series(t).groupby(gid).transform("first").to_numpy()
    cum_sec = (t - t0).astype(float) / 1e9
    cum_ground = _group_cumsum(np.nan_to_num(step), gid)
    cum_max = pd.Series(s).groupby(gid).cummax().to_numpy()
    net = np.hypot(x - pd.Series(x).groupby(gid).transform("first").to_numpy(),
                   y - pd.Series(y).groupby(gid).transform("first").to_numpy())
    cum_abs_turn = _group_cumsum(np.abs(np.nan_to_num(d_dir)), gid)
    cum_net_turn = _group_cumsum(np.nan_to_num(d_dir), gid)
    f["cum_seconds"], f["cum_ground"], f["cum_max_speed"] = cum_sec, cum_ground, cum_max
    f["cum_angular_dist"], f["cum_net_turn"] = cum_abs_turn, cum_net_turn
    f["turn_wiggle"] = cum_abs_turn - np.abs(cum_net_turn)
    f["net_disp"] = net
    with np.errstate(invalid="ignore", divide="ignore"):
        f["straightness"] = np.where(cum_ground > 0.5, net / cum_ground, np.nan)
        f["tortuosity"] = np.where(net > 0.5, cum_ground / net, np.nan)
    at_max = np.isfinite(s) & (s >= cum_max)
    f["t_at_cum_max"] = pd.Series(np.where(at_max, cum_sec, np.nan)).groupby(gid).ffill().to_numpy()
    f["dist_at_cum_max"] = pd.Series(np.where(at_max, cum_ground, np.nan)).groupby(gid).ffill().to_numpy()

    # rolling window
    k = cfg["roll_frames"]
    mp = max(k // 2, 2)
    _, _, roll_path, _ = _rolling(np.nan_to_num(step, nan=np.nan), within, k, mp)
    f["roll_path"] = roll_path
    x_lag, y_lag = _gshift(x, within, gid, k - 1), _gshift(y, within, gid, k - 1)
    f["roll_net"] = np.where(np.isfinite(x_lag), np.hypot(x - x_lag, y - y_lag), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["roll_straightness"] = np.where(roll_path > 0.5, f["roll_net"] / roll_path, np.nan)
    sm, ss, _, _ = _rolling(s, within, k, mp)
    f["roll_speed_mean"], f["roll_speed_std"] = sm, ss
    f["speed_dev_roll"] = s - sm
    f["roll_abs_omega"], _, _, _ = _rolling(np.abs(omega), within, k, mp)
    f["ewma_speed"] = _ewma(s, gid, 2.0 / (cfg["ewma_span"] + 1))

    # constant-velocity / constant-acceleration position forecast error
    vxs, vys = f["vx"].to_numpy(), f["vy"].to_numpy()
    ax_, ay_ = _gdiff(vxs, within) / dt, _gdiff(vys, within) / dt
    for h in cfg["forecast_horizons"]:
        xf, yf = _gshift(x, within, gid, -h), _gshift(y, within, gid, -h)
        tau = h * 0.1
        f[f"fc_cv_err_{h}"] = np.hypot(xf - (x + vxs * tau), yf - (y + vys * tau))
        f[f"fc_ca_err_{h}"] = np.hypot(xf - (x + vxs * tau + 0.5 * ax_ * tau**2), yf - (y + vys * tau + 0.5 * ay_ * tau**2))

    if "event" in df:
        snap = (df["event"] == "ball_snap").to_numpy()
        t_snap = pd.Series(np.where(snap, cum_sec, np.nan)).groupby(gid).transform("max").to_numpy()
        f["t_since_snap"] = cum_sec - t_snap

    f["row"] = np.arange(len(df))
    return f.astype({c: "float32" for c in f.columns if f[c].dtype == "float64"})


def _group_cumsum(v: np.ndarray, gid: np.ndarray) -> np.ndarray:
    return pd.Series(v).groupby(gid).cumsum().to_numpy()


# ---------------------------------------------------------------------------------------------------------------
# change-of-direction events
# ---------------------------------------------------------------------------------------------------------------

def detect_cuts_one(t: np.ndarray, s: np.ndarray, dirn: np.ndarray, d_dir: np.ndarray, omega: np.ndarray, step: np.ndarray,
                    cfg: dict) -> list[dict]:
    """Cuts in one trajectory: peaks of the smoothed |turn rate|, with a window, speeds, angle and COD deficit.

    A reversal through a stop (5-10-5 plants) has its turn-rate peak at near-zero speed, where `dir` is unreliable, so
    the movement gate is on the speed before AND after the turn, not at the peak. That also rejects direction jitter at
    the start or end of a run, where one side is still. The angle is exit heading minus entry heading.
    """
    n = len(t)
    if n < 5:
        return []
    w = np.abs(np.nan_to_num(omega))
    sm = np.convolve(w, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(sm, height=cfg["cut_omega_thresh"], distance=cfg["cut_min_separation"])
    c = cfg["cut_context_frames"]
    cuts = []
    for p in peaks:
        thr = cfg["cut_window_frac"] * sm[p]
        lo, hi = p, p
        while lo - 1 >= 0 and sm[lo - 1] > thr:
            lo -= 1
        while hi + 1 < n and sm[hi + 1] > thr:
            hi += 1
        pre = slice(max(lo - c, 0), lo + 1)
        post = slice(hi, min(hi + c + 1, n))
        ctx = slice(max(lo - c, 0), min(hi + c + 1, n))
        if not (np.isfinite(s[pre]).any() and np.isfinite(s[post]).any()):
            continue
        entry_i = pre.start + int(np.nanargmax(s[pre]))
        entry = float(s[entry_i])
        exit_i = post.start + int(np.nanargmax(s[post]))
        exit_ = float(s[exit_i])
        if min(entry, exit_) < cfg["cut_min_speed"]:
            continue
        seg = slice(entry_i, exit_i + 1)
        min_i = entry_i + int(np.nanargmin(s[seg]))
        dist_ctx = float(np.nansum(step[ctx.start + 1:ctx.stop]))
        time_ctx = float(t[ctx.stop - 1] - t[ctx.start])
        angle = float(wrap_deg(dirn[exit_i] - dirn[entry_i]))
        t_in = float(t[hi] - t[lo]) + 0.1
        decel_dt = float(t[min_i] - t[entry_i])
        cuts.append({
            "peak_idx": int(p), "start_idx": int(lo), "end_idx": int(hi),
            "t_peak": float(t[p]), "side": "R" if angle > 0 else "L", "cut_angle": angle,
            "reversal": bool(abs(angle) > cfg["reversal_deg"]), "peak_omega": float(sm[p]),
            "time_in_cut": t_in, "entry_speed": entry, "min_speed": float(s[min_i]), "exit_speed": exit_,
            "retention": exit_ / entry if entry > 0 else np.nan,
            "decel_dist_in": float(np.nansum(step[entry_i + 1:min_i + 1])),
            "decel_rate_in": (entry - float(s[min_i])) / decel_dt if decel_dt > 0 else np.nan,
            "reaccel_time_out": float(t[exit_i] - t[min_i]),
            # time through the cut window vs carrying the entry speed straight over the same distance
            "cod_deficit": time_ctx - dist_ctx / entry if entry > 0 else np.nan,
        })
    return cuts


# ---------------------------------------------------------------------------------------------------------------
# trajectory-level metrics
# ---------------------------------------------------------------------------------------------------------------

def sparc(speed: np.ndarray, fs: float = 10.0, pad: int = 4, fc: float = 4.0, amp_th: float = 0.05) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2012). More negative = less smooth."""
    v = speed[np.isfinite(speed)]
    if len(v) < 8 or np.ptp(v) == 0:
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(v))) + pad))
    f = rfftfreq(nfft, 1 / fs)
    m = np.abs(rfft(v, nfft))
    m = m / m.max()
    sel = f <= fc
    f, m = f[sel], m[sel]
    idx = np.flatnonzero(m >= amp_th)
    f, m = f[idx[0]:idx[-1] + 1], m[idx[0]:idx[-1] + 1]
    if len(f) < 2:
        return np.nan
    fn = (f - f[0]) / (f[-1] - f[0])
    return float(-np.sum(np.sqrt(np.diff(fn) ** 2 + np.diff(m) ** 2)))


def log_dimensionless_jerk(speed: np.ndarray, dt: float = 0.1) -> float:
    """-ln( T^3 / v_peak^2 * integral of (d^2 v / dt^2)^2 dt ). Higher = smoother."""
    v = speed[np.isfinite(speed)]
    if len(v) < 6 or v.max() <= 0:
        return np.nan
    j = np.gradient(np.gradient(v, dt), dt)
    T = dt * (len(v) - 1)
    integral = np.sum(j**2) * dt
    return float(-np.log(T**3 / v.max() ** 2 * integral)) if integral > 0 else np.nan


def spectral_features(speed: np.ndarray, fs: float = 10.0) -> tuple[float, float]:
    """(dominant non-DC frequency in Hz, normalized spectral entropy) of the mean-removed speed."""
    v = speed[np.isfinite(speed)]
    if len(v) < 8 or np.ptp(v) == 0:
        return np.nan, np.nan
    p = np.abs(rfft(v - v.mean())) ** 2
    f = rfftfreq(len(v), 1 / fs)
    p, f = p[1:], f[1:]
    if p.sum() <= 0:
        return np.nan, np.nan
    q = p / p.sum()
    ent = -np.sum(q[q > 0] * np.log(q[q > 0])) / np.log(len(q)) if len(q) > 1 else np.nan
    return float(f[np.argmax(p)]), float(ent)


def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < m + 5 or x.std() == 0:
        return np.nan
    r = r_frac * x.std()

    def count(mm):
        T = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        d = np.max(np.abs(T[:, None, :] - T[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(T)) / 2

    b, a = count(m), count(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else np.nan


def _autocorr1(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if len(v) < 5 or v.std() == 0:
        return np.nan
    v = v - v.mean()
    return float(np.sum(v[1:] * v[:-1]) / np.sum(v * v))


def _lag_o_vs_dir(omega_o: np.ndarray, omega_d: np.ndarray, max_lag: int = 10, dt: float = 0.1) -> float:
    """Lag (s) at which the turn rate of `dir` best matches that of `o`; positive = body orientation leads the path."""
    ok = np.isfinite(omega_o) & np.isfinite(omega_d)
    if ok.sum() < 3 * max_lag:
        return np.nan
    a, b = np.where(ok, omega_o, 0.0), np.where(ok, omega_d, 0.0)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        u, w = (a[:len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:lag])
        if len(u) > 5 and u.std() > 0 and w.std() > 0:
            r = np.corrcoef(u, w)[0, 1]
            if r > best:
                best, best_lag = r, lag
    return best_lag * dt


def trajectory_metrics(df: pd.DataFrame, f: pd.DataFrame, keys: list[str], cfg: dict, cuts: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per trajectory. `df` is the frame table and `f` the output of add_frame_features (same order)."""
    gid, _, _ = _group_layout(df, keys)
    edges = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
    x, y, s = df["x"].to_numpy(float), df["y"].to_numpy(float), df["s"].to_numpy(float)
    a_long, omega = f["a_long"].to_numpy(float), f["omega"].to_numpy(float)
    step, d_dir = f["step"].to_numpy(float), f["d_dir"].to_numpy(float)
    cum_sec, cum_ground = f["cum_seconds"].to_numpy(float), f["cum_ground"].to_numpy(float)
    has_o = "o_offset" in f
    off = f["o_offset"].to_numpy(float) if has_o else None
    om_o = f["omega_o"].to_numpy(float) if has_o else None
    events = df["event"].to_numpy() if "event" in df else None
    first = df.iloc[edges[:-1]][keys + [c for c in ("nfl_id", "drill_type", "drill_name", "attempt", "draft_year") if c in df and c not in keys]]
    cut_by_traj = cuts.groupby("_gid") if cuts is not None and len(cuts) else None

    rows = []
    for g, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sl = slice(a, b)
        xs, ys, ss = x[sl], y[sl], s[sl]
        n = b - a
        r = {"n_frames": n, "duration": float(cum_sec[b - 1]), "n_gaps": int(f["dt_gap"].to_numpy()[sl].sum())}
        path = float(cum_ground[b - 1])
        net = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        r.update(path_length=path, net_disp=net, straightness=net / path if path > 0.5 else np.nan,
                 tortuosity=path / net if net > 0.5 else np.nan)
        ab = float(np.nansum(np.abs(d_dir[sl])))
        nt = float(np.nansum(d_dir[sl]))
        r.update(abs_turn=ab, net_turn=nt, turn_wiggle=ab - abs(nt))
        if net > 0.5:
            chord = np.array([xs[-1] - xs[0], ys[-1] - ys[0]]) / net
            rel = np.c_[xs - xs[0], ys - ys[0]]
            r["max_perp_dev"] = float(np.max(np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0])))
        else:
            r["max_perp_dev"] = np.nan
        try:
            r["hull_area"] = float(ConvexHull(np.c_[xs, ys]).volume) if n >= 3 else np.nan
        except Exception:
            r["hull_area"] = 0.0
        r["radius_gyration"] = float(np.sqrt(np.mean((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)))

        smax = float(np.nanmax(ss)) if np.isfinite(ss).any() else np.nan
        r["max_speed"], r["mean_speed"] = smax, float(np.nanmean(ss))
        if np.isfinite(smax) and smax > 0:
            im = int(np.nanargmax(ss))
            r["time_to_max"], r["dist_to_max"] = float(cum_sec[a + im]), float(cum_ground[a + im])
            for pct in (50, 80, 90):
                hit = np.flatnonzero(ss >= pct / 100 * smax)
                r[f"time_to_{pct}pct"] = float(cum_sec[a + hit[0]])
            r["frames_95pct_max"] = int((ss >= 0.95 * smax).sum())
            r["decay_after_peak"] = float((smax - ss[-1]) / smax)
        al = a_long[sl]
        r["max_accel"] = float(np.nanmax(al)) if np.isfinite(al).any() else np.nan
        r["max_decel"] = float(np.nanmin(al)) if np.isfinite(al).any() else np.nan
        r["mean_abs_a_long"] = float(np.nanmean(np.abs(al))) if np.isfinite(al).any() else np.nan
        jerk = f["jerk_long"].to_numpy(float)[sl]
        r["jerk_rms"] = float(np.sqrt(np.nanmean(jerk**2))) if np.isfinite(jerk).any() else np.nan
        r["ldlj"], r["sparc"] = log_dimensionless_jerk(ss), sparc(ss)
        r["speed_dom_freq"], r["speed_spec_entropy"] = spectral_features(ss)
        r["a_long_autocorr1"] = _autocorr1(al)
        om = omega[sl]
        r["omega_var"] = float(np.nanvar(om)) if np.isfinite(om).any() else np.nan
        r["mean_abs_omega"] = float(np.nanmean(np.abs(om))) if np.isfinite(om).any() else np.nan
        r["sampen_dir"] = sample_entropy(d_dir[sl]) if n >= cfg["min_frames_entropy"] else np.nan

        stat = ss < cfg["stationary_speed"]
        r["stationary_jitter"] = float(np.nanmean(step[sl][stat])) if (stat & np.isfinite(step[sl])).any() else np.nan
        r["speed_resid_mad"] = float(np.nanmedian(np.abs(f["speed_resid"].to_numpy(float)[sl])))
        r["dis_over_step_median"] = float(np.nanmedian(f["dis_over_step"].to_numpy(float)[sl])) if np.isfinite(f["dis_over_step"].to_numpy(float)[sl]).any() else np.nan

        if has_o:
            of = off[sl]
            okf = np.isfinite(of)
            if okf.any():
                r["frac_forward"] = float((np.abs(of[okf]) < cfg["band_forward_deg"]).mean())
                r["frac_back"] = float((np.abs(of[okf]) > cfg["band_back_deg"]).mean())
                r["frac_side"] = 1.0 - r["frac_forward"] - r["frac_back"]
                r["mean_abs_o_offset"] = float(np.abs(of[okf]).mean())
                r["o_dir_lag"] = _lag_o_vs_dir(om_o[sl], om)
        if events is not None:
            ev = events[sl]
            i_snap = np.flatnonzero(ev == "ball_snap")
            i_pf = np.flatnonzero(ev == "pass_forward")
            if len(i_snap):
                r["speed_at_snap"] = float(ss[i_snap[0]])
            if len(i_pf):
                r["speed_at_pass_forward"] = float(ss[i_pf[0]])
            if len(i_snap) and len(i_pf) and i_pf[0] >= i_snap[0]:
                r["dist_snap_to_pass_forward"] = float(cum_ground[a + i_pf[0]] - cum_ground[a + i_snap[0]])
                r["time_snap_to_pass_forward"] = float(cum_sec[a + i_pf[0]] - cum_sec[a + i_snap[0]])

        c = cut_by_traj.get_group(g) if cut_by_traj is not None and g in cut_by_traj.groups else None
        nc = 0 if c is None else len(c)
        side = c[~c["reversal"]] if c is not None else None      # a reversal through a stop has no meaningful side
        nl = 0 if c is None else int((side["side"] == "L").sum())
        nr = 0 if c is None else int((side["side"] == "R").sum())
        r.update(n_cuts=nc, cuts_per_sec=nc / r["duration"] if r["duration"] > 0 else np.nan,
                 n_reversals=0 if c is None else int(c["reversal"].sum()), n_cuts_left=nl, n_cuts_right=nr,
                 cut_asymmetry=(nr - nl) / (nr + nl) if nr + nl else np.nan)
        if nc:
            for col in ("entry_speed", "min_speed", "exit_speed", "retention", "time_in_cut", "decel_rate_in",
                        "reaccel_time_out", "cod_deficit"):
                r[f"cut_{col}_mean"] = float(c[col].mean())
            r["cut_angle_abs_mean"] = float(c["cut_angle"].abs().mean())
            r["cut_angle_abs_max"] = float(c["cut_angle"].abs().max())
        rows.append(r)

    out = pd.DataFrame(rows, index=first.index)
    return pd.concat([first, out], axis=1).reset_index(drop=True)


def cuts_table(df: pd.DataFrame, f: pd.DataFrame, keys: list[str], cfg: dict) -> pd.DataFrame:
    """All cuts, with their trajectory keys and the global row index of the peak frame (`row`)."""
    gid, _, _ = _group_layout(df, keys)
    edges = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
    d_dir, omega, step, cum_sec = (f[c].to_numpy(float) for c in ("d_dir", "omega", "step", "cum_seconds"))
    sp, dirn = df["s"].to_numpy(float), df["dir"].to_numpy(float)
    rows = []
    for g, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        for c in detect_cuts_one(cum_sec[a:b], sp[a:b], dirn[a:b], d_dir[a:b], omega[a:b], step[a:b], cfg):
            c["_gid"] = g
            for k in ("peak_idx", "start_idx", "end_idx"):
                c[k.replace("_idx", "_row")] = a + c.pop(k)
            rows.append(c)
    cuts = pd.DataFrame(rows)
    if len(cuts):
        first = df.iloc[edges[:-1]][keys].reset_index(drop=True)
        cuts = pd.concat([first.iloc[cuts["_gid"]].reset_index(drop=True), cuts], axis=1)
    return cuts


def add_cut_timing(f: pd.DataFrame, df: pd.DataFrame, keys: list[str], cuts: pd.DataFrame) -> pd.DataFrame:
    """Per-frame `t_since_cut` and `n_cuts_so_far`."""
    gid, _, _ = _group_layout(df, keys)
    marker = np.full(len(df), np.nan)
    counter = np.zeros(len(df))
    if len(cuts):
        marker[cuts["peak_row"].to_numpy()] = f["cum_seconds"].to_numpy(float)[cuts["peak_row"].to_numpy()]
        counter[cuts["peak_row"].to_numpy()] = 1.0
    last = pd.Series(marker).groupby(gid).ffill().to_numpy()
    f = f.copy()
    f["t_since_cut"] = (f["cum_seconds"].to_numpy(float) - last).astype("float32")
    f["n_cuts_so_far"] = _group_cumsum(counter, gid).astype("float32")
    return f


# ---------------------------------------------------------------------------------------------------------------
# combine-only post-processing
# ---------------------------------------------------------------------------------------------------------------

REPEAT_COLS = ["max_speed", "path_length", "duration", "max_decel", "n_cuts", "cut_retention_mean"]


def attempt_repeatability(traj: pd.DataFrame, cols=REPEAT_COLS) -> pd.DataFrame:
    """Per player x drill: change from the previous attempt, SD across attempts, and attempt rank."""
    traj = traj.sort_values(["nfl_id", "drill_name", "attempt"]).copy()
    g = traj.groupby(["nfl_id", "drill_name"])
    traj["attempt_rank"] = g.cumcount() + 1
    for c in cols:
        if c in traj:
            traj[f"{c}_delta_prev"] = g[c].diff()
            traj[f"{c}_sd_attempts"] = g[c].transform("std")
            traj[f"{c}_minus_player_mean"] = traj[c] - g[c].transform("mean")
    return traj.sort_index()


def add_normalized(traj: pd.DataFrame, results: pd.DataFrame, z_cols=("max_speed", "duration", "path_length")) -> pd.DataFrame:
    """Size-normalized versions, Froude number and z-scores within drill x position x year (combine only)."""
    res = results[["nfl_id", "combine_height", "combine_weight", "wing_span", "combine_position"]]
    out = traj.merge(res, on="nfl_id", how="left")
    h_m = out["combine_height"] * 0.0254
    out["froude_max"] = (out["max_speed"] * YD_TO_M) ** 2 / (G * h_m)
    out["max_speed_per_height"] = out["max_speed"] * YD_TO_M / h_m
    out["path_length_per_wingspan"] = out["path_length"] / (out["wing_span"] * 0.0254 / YD_TO_M)
    grp = ["drill_name", "combine_position", "draft_year"]
    for c in z_cols:
        g = out.groupby(grp)[c]
        out[f"{c}_z"] = (out[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------------------------------------------

def compute(df: pd.DataFrame, keys: list[str], cfg: dict):
    """Frame features, cuts and trajectory metrics for a sorted frame table. Returns (frames, traj, cuts)."""
    df = df.sort_values(keys + ["time"], kind="stable").reset_index(drop=True)
    f = add_frame_features(df, keys, cfg)
    cuts = cuts_table(df, f, keys, cfg)
    f = add_cut_timing(f, df, keys, cuts)
    traj = trajectory_metrics(df, f, keys, cfg, cuts)
    frames = pd.concat([df[keys + ["time"] + [c for c in ("nfl_id", "drill_type", "drill_name", "attempt", "x", "y", "s", "a", "dir", "o") if c in df and c not in keys]], f], axis=1)
    return frames, traj, cuts.drop(columns=["_gid"], errors="ignore")


def run_game(seasons, write_frames: bool, cfg: dict) -> None:
    for season in seasons:
        src = PROCESSED / f"game_tracking_std_{season}.parquet"
        print(f"{season}: reading {src.name}")
        df = pd.read_parquet(src)
        frames, traj, cuts = compute(df, GAME_KEYS, cfg)
        _write(f"game_metrics_{season}", frames if write_frames else None, traj, cuts)


def run_combine(write_frames: bool, cfg: dict) -> None:
    from loaders import load_results

    raw = pd.read_csv(ROOT / "raw" / "combine_tracking.csv")
    df = raw[raw["entity_type"] == "PLAYER"].copy()
    df["time"] = pd.to_datetime(df["time"])
    frames, traj, cuts = compute(df, COMBINE_KEYS, cfg)
    traj = attempt_repeatability(traj)
    traj = add_normalized(traj, load_results())
    _write("combine_metrics", frames if write_frames else None, traj, cuts)


def _write(prefix: str, frames, traj, cuts) -> None:
    PROCESSED.mkdir(exist_ok=True)
    if frames is not None:
        frames.to_parquet(PROCESSED / f"{prefix}_frames.parquet", index=False)
    traj.to_parquet(PROCESSED / f"{prefix}_traj.parquet", index=False)
    cuts.to_parquet(PROCESSED / f"{prefix}_cuts.parquet", index=False)
    print(f"  wrote {prefix}: {len(traj):,} trajectories, {len(cuts):,} cuts" + (f", {len(frames):,} frames" if frames is not None else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["game", "combine"], default="game")
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--no-frames", action="store_true", help="skip the (large) frame-level output")
    args = ap.parse_args()
    cfg = pipeline_cfg()["metrics"]
    if args.source == "combine":
        run_combine(not args.no_frames, cfg)
    else:
        run_game(args.seasons, not args.no_frames, cfg)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------
# original feature brainstorm, kept for reference (see the docstring at the top for what is implemented)
# ---------------------------------------------------------------------------------------------------------------
BRAINSTORM = '''
Here's a broad set, grouped by what it measures. Unless noted, Δt = 0.1 s.

**Per-frame kinematics (derived from the raw columns)**
- Euclidean step: √(Δx² + Δy²) from consecutive frames. The ratio of `dis` to Euclidean step is near 1 on straight running and rises with in-frame curvature or tracking jitter.
- Velocity components: vx, vy from `s` and `dir`. Check the angle convention; in BDB, 0° usually points along +y and angles run clockwise.
- Signed longitudinal acceleration: Δs/Δt. The provided `a` is usually a magnitude, so it can't tell acceleration from deceleration.
- Lateral (centripetal) acceleration: s · ω_dir. Comparing it to longitudinal acceleration shows how much effort goes into turning versus speeding up.
- Jerk: Δa/Δt, plus the jerk of signed acceleration.
- Turning:
  - Angular velocity and acceleration of `dir`, and of `o` where it exists.
  - Always wrap angle differences with `atan2(sin Δ, cos Δ)`.
  - Curvature κ = Δdir / step length, and turn radius = 1/κ.
- Orientation offset: o − dir (wrapped).
  - About 0° is running forward, about ±90° is shuffling or crabbing, and about 180° is backpedaling.
  - Also track time spent in each band.
- Consistency check: reported `s` versus speed computed from position. The residual works as a noise or quality flag.

**Path geometry (cumulative and rolling-window)**
- Cumulative path length, net displacement, and their ratio (straightness index). The inverse of that ratio is tortuosity.
- All of these over rolling k-frame windows as well as cumulatively.
- Total absolute turning versus net turning. The gap between them measures wiggle.
- Maximum perpendicular deviation from the start-to-end chord.
- Convex hull area of the path, and radius of gyration around its centroid.
- Fractal dimension or box-counting of the path. This is more useful on long game paths.
- Efficiency against the drill's ideal path, since combine cone layouts are known.

**Speed profile**
- Max speed, time to max, distance to max, and time to reach 50%, 80%, or 90% of max.
- Splits computed from tracking (0–10, 10–20, 20–40, flying 10). These can be checked against the official splits.
- Sprint force-velocity profile: fit v(t) = Vmax · (1 − e^(−t/τ)) to the 40-yard dash.
  - From the fit you get F0, V0, Pmax, and the slope of the F-v line.
  - Multiplying by `combine_weight` gives real force and power. This is probably the single best 40-yard dash feature.
- Speed sustain: frames spent within 95% of max, and the decay after the peak.
- Energy terms: momentum m·v, kinetic energy ½mv², and instantaneous power m·a·v.

**Change-of-direction events** (detect cuts as peaks in |ω| or κ)
- Entry speed, minimum speed, exit speed, and retention (exit ÷ entry).
- Cut angle, time spent in the cut, deceleration distance and rate going in, and re-acceleration time coming out.
- COD deficit: time through the cut minus a linear sprint over the same distance.
- Left versus right asymmetry. This is directly measurable in the 3-cone, shuttle, and DB transition drills.
- Time since the last cut, and cuts per second.

**Smoothness and complexity**
- Log dimensionless jerk and SPARC (spectral arc length). Both are standard motor-control smoothness metrics.
- FFT of speed and acceleration: dominant frequency and spectral entropy.
- Sample entropy or approximate entropy of `dir` and of acceleration.
- Autocorrelation of acceleration, and variance of turning rate.

**Body orientation (game tracking only; `combine_tracking` has no `o`)**
- Hip-flip time: how long the large swing in `o` takes during DB transitions.
- Duration and speed of backpedaling.
- Lag between orientation and direction, i.e. whether the body turns before or after the path does.
- Orientation relative to the line of scrimmage, the sideline, or the target endzone.

**Normalized and anthropometric**
- Per-weight, per-height, and per-wingspan versions of any of the above.
- Froude number s² / (g · height) for size-scaled speed.
- Z-scores within drill × position × year.
- Combine-to-game translation ratios: game max speed ÷ combine max speed, and game COD retention ÷ combine COD retention.

**Repeatability across attempts**
- Variance across attempts of the same drill.
- Change from attempt 1 to attempt 2, which can reflect learning or fatigue.
- Best attempt minus mean attempt.

**Event-anchored context (game)**
- Standardize `play_direction` first so all plays run the same way.
- Frames since `ball_snap`, speed at `pass_forward`, and distance covered from snap to throw.
- Get-off computed from tracking, compared against `player_get_off`.
- Position relative to the line of scrimmage (from `yardline_side` and `yardline_number`) and relative to the sideline.
- Caveat: game tracking appears to cover only rookies, not all 22 players or the ball. Relational features such as nearest defender or closing speed only work when two rookies share a play, so check coverage before planning around them.

**Prediction residuals (useful for next-k-frame modeling)**
- Error of a constant-velocity or constant-acceleration forecast against actual position, 1 to k frames ahead. This acts as a "surprise" or unpredictability feature per player.
- Kalman-filtered state, its innovation, and innovation magnitude over time.
- Lagged values, rolling mean and standard deviation, EWMA, and deviation from the rolling mean.

**Data quality flags**
- Time gaps that aren't 0.1 s (dropped frames).
- Position jitter while stationary.
- The `dis` versus Euclidean mismatch and the speed residual from above.
'''