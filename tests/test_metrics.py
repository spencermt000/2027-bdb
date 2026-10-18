import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import ROOT, pipeline_cfg

spec = importlib.util.spec_from_file_location("compute_metrics", Path(ROOT) / "src" / "process" / "02_compute_metrics.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
CFG = pipeline_cfg()["metrics"]
KEYS = ["event_id"]


def track(event, x, y, dt=0.1, nfl_id=1, t0="2024-03-01 12:00:00"):
    """Frames from position arrays; s, dir (clockwise from +y) and a are derived from the positions."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    vx, vy = np.gradient(x, dt), np.gradient(y, dt)
    s = np.hypot(vx, vy)
    return pd.DataFrame({"event_id": event, "nfl_id": nfl_id, "time": pd.Timestamp(t0) + pd.to_timedelta(np.arange(n) * dt, unit="s"),
                         "x": x, "y": y, "s": s, "a": np.abs(np.gradient(s, dt)), "dis": 0.0,
                         "dir": np.degrees(np.arctan2(vx, vy)) % 360})


def run(df):
    return cm.compute(df, KEYS, CFG)


def test_wrap_deg():
    assert cm.wrap_deg(190) == pytest.approx(-170) and cm.wrap_deg(-190) == pytest.approx(170)
    assert cm.wrap_deg(359 - 1) == pytest.approx(-2) and cm.wrap_deg(0) == 0


def test_straight_run_kinematics_and_conventions():
    t = np.arange(0, 3, 0.1)
    df = track("a", 10 + 2 * t + 1.5 * t**2, np.full(len(t), 5.0))      # +x, constant accel 3 yd/s^2
    frames, traj, cuts = run(df)
    assert np.allclose(frames["dir"].iloc[2:-2], 90.0, atol=1)           # +x is 90 deg clockwise from +y
    assert np.allclose(frames["vx"].iloc[2:-2], frames["s"].iloc[2:-2], atol=0.2)
    assert np.allclose(frames["vy"].iloc[2:-2], 0.0, atol=0.1)
    assert np.allclose(frames["a_long"].iloc[3:-2], 3.0, atol=0.3)
    assert np.allclose(frames["step"].iloc[1:], np.hypot(np.diff(df["x"]), 0.0), atol=1e-5)
    r = traj.iloc[0]
    assert r["path_length"] == pytest.approx(r["net_disp"], rel=1e-5) and r["straightness"] == pytest.approx(1.0)
    assert r["max_perp_dev"] == pytest.approx(0.0, abs=1e-5) and r["hull_area"] == pytest.approx(0.0, abs=1e-5)
    assert r["n_cuts"] == 0 and r["time_to_max"] == pytest.approx(t[-1], abs=0.11)
    assert frames["cum_seconds"].iloc[0] == 0 and frames["cum_seconds"].iloc[-1] == pytest.approx(2.9, abs=1e-4)
    assert frames["cum_ground"].iloc[-1] == pytest.approx(df["x"].iloc[-1] - df["x"].iloc[0], rel=1e-4)


def test_circle_turning_and_lateral_acceleration():
    R, speed, n = 5.0, 5.0, 40
    th = np.arange(n) * 0.1 * speed / R                                  # clockwise circle
    df = track("c", R * np.sin(th), R * np.cos(th))
    frames, traj, _ = run(df)
    mid = frames.iloc[5:-5]
    assert np.allclose(mid["omega"], np.degrees(speed / R), rtol=0.05)   # clockwise = positive
    assert np.allclose(mid["a_lat"], speed**2 / R, rtol=0.1)
    assert np.allclose(mid["turn_radius"], R, rtol=0.1)
    assert traj.iloc[0]["abs_turn"] == pytest.approx(abs(traj.iloc[0]["net_turn"]), rel=1e-4)   # no wiggle in a smooth arc
    assert traj.iloc[0]["radius_gyration"] > 0 and traj.iloc[0]["hull_area"] > 0


def test_reversal_through_a_stop_is_one_cut_with_correct_speeds():
    """Run +x at 5 yd/s, brake to a stop, reverse to -x at 5 yd/s: the 5-10-5 plant pattern."""
    dt = 0.1
    v = np.r_[np.full(15, 5.0), np.linspace(5, 0, 6)[1:], np.linspace(0, -5, 6)[1:], np.full(15, -5.0)]
    x = 20 + np.cumsum(v * dt)
    df = track("r", x, np.full(len(x), 10.0))
    df["dir"] = np.where(df["s"] < 0.3, 90.0, np.where(v > 0, 90.0, 270.0))   # `dir` is undefined at rest: keep it noisy-free
    frames, traj, cuts = run(df)
    assert len(cuts) == 1
    c = cuts.iloc[0]
    assert c["reversal"] and abs(c["cut_angle"]) == pytest.approx(180, abs=1)
    assert c["entry_speed"] == pytest.approx(5.0, abs=0.3) and c["exit_speed"] == pytest.approx(5.0, abs=0.3)
    assert c["min_speed"] < 1.0 and c["retention"] == pytest.approx(1.0, abs=0.1)
    assert c["decel_rate_in"] > 3 and c["cod_deficit"] > 0
    r = traj.iloc[0]
    assert r["n_cuts"] == 1 and r["n_reversals"] == 1 and r["n_cuts_left"] + r["n_cuts_right"] == 0
    assert frames["n_cuts_so_far"].iloc[-1] == 1 and np.isnan(frames["t_since_cut"].iloc[0])
    assert frames["t_since_cut"].iloc[-1] > 0


def test_cut_side_and_asymmetry():
    """A 90-degree turn mid-run: clockwise (right) from +y heading to +x."""
    dt = 0.1
    heading = np.r_[np.zeros(15), np.linspace(0, 90, 4)[1:], np.full(15, 90.0)]
    th = np.radians(heading)
    x, y = np.cumsum(5 * np.sin(th) * dt), np.cumsum(5 * np.cos(th) * dt)
    df = track("s", x, y)
    df["dir"] = heading
    _, traj, cuts = run(df)
    assert len(cuts) == 1 and cuts.iloc[0]["side"] == "R" and not cuts.iloc[0]["reversal"]
    assert cuts.iloc[0]["cut_angle"] == pytest.approx(90, abs=5)
    assert traj.iloc[0]["n_cuts_right"] == 1 and traj.iloc[0]["cut_asymmetry"] == 1.0


def test_start_and_end_of_run_jitter_is_not_a_cut():
    """Standing with noisy heading, then sprinting off, then stopping: no cuts."""
    rng = np.random.default_rng(0)
    v = np.r_[np.zeros(8), np.linspace(0, 8, 8), np.full(12, 8.0), np.linspace(8, 0, 8), np.zeros(6)]
    x = np.cumsum(v * 0.1)
    df = track("j", x, np.zeros(len(x)))
    df["dir"] = np.where(df["s"] < 0.3, rng.uniform(0, 360, len(df)), 90.0)
    _, traj, cuts = run(df)
    assert len(cuts) == 0 and traj.iloc[0]["n_cuts"] == 0


def test_trajectories_do_not_leak_across_boundaries():
    t = np.arange(0, 2, 0.1)
    a = track("a", 10 + 5 * t, np.zeros(len(t)), nfl_id=1)
    b = track("b", 80 - 3 * t, np.full(len(t), 20.0), nfl_id=2)
    frames, traj, _ = run(pd.concat([a, b], ignore_index=True))
    fb = frames[frames["event_id"] == "b"]
    assert np.isnan(fb["step"].iloc[0]) and np.isnan(fb["a_long"].iloc[0]) and fb["cum_ground"].iloc[0] == 0
    assert np.isnan(fb["roll_net"].iloc[:CFG["roll_frames"] - 1]).all()
    last_a = frames[frames["event_id"] == "a"].iloc[-1]
    assert np.isnan(last_a["fc_cv_err_1"]) and np.isnan(last_a["fc_cv_err_5"])
    assert set(traj["event_id"]) == {"a", "b"} and (traj["n_frames"] == 20).all()
    assert fb["cum_max_speed"].iloc[0] == pytest.approx(fb["s"].iloc[0])


def test_rolling_helper_matches_pandas_within_groups():
    rng = np.random.default_rng(1)
    v = rng.normal(size=60)
    v[7] = np.nan
    gid = np.repeat([0, 1, 2], [25, 20, 15])
    within = np.r_[np.arange(25), np.arange(20), np.arange(15)]
    mean, std, _, _ = cm._rolling(v, within, k=5, min_periods=3)
    exp_mean = pd.Series(v).groupby(gid).transform(lambda s: s.rolling(5, min_periods=3).mean()).to_numpy()
    exp_std = pd.Series(v).groupby(gid).transform(lambda s: s.rolling(5, min_periods=3).std()).to_numpy()
    assert np.allclose(mean, exp_mean, equal_nan=True, atol=1e-9)
    assert np.allclose(std, exp_std, equal_nan=True, atol=1e-6)


def test_ewma_resets_per_trajectory():
    v = np.r_[np.full(5, 1.0), np.full(5, 9.0)]
    gid = np.r_[np.zeros(5, int), np.ones(5, int)]
    out = cm._ewma(v, gid, 0.5)
    assert np.allclose(out[:5], 1.0) and np.allclose(out[5:], 9.0)


def test_forecast_errors_zero_for_constant_velocity_and_acceleration():
    t = np.arange(0, 3, 0.1)
    cv = track("cv", 5 * t, 2 * t)
    ca = track("ca", 0.5 * 2 * t**2, np.zeros(len(t)))
    fcv, _, _ = run(cv)
    fca, _, _ = run(ca)
    assert np.nanmax(fcv["fc_cv_err_5"].iloc[2:-6]) < 0.05
    assert np.nanmax(fca["fc_ca_err_5"].iloc[3:-6]) < np.nanmax(fca["fc_cv_err_5"].iloc[3:-6])


def test_smoothness_metrics_rank_smooth_above_jerky():
    t = np.arange(0, 4, 0.1)
    smooth = 8 * np.sin(np.pi * t / 4) ** 2
    jerky = smooth + 0.8 * np.sign(np.sin(2 * np.pi * 3 * t))
    assert cm.sparc(smooth) > cm.sparc(jerky)
    assert cm.log_dimensionless_jerk(smooth) > cm.log_dimensionless_jerk(jerky)
    f_dom, ent = cm.spectral_features(np.sin(2 * np.pi * 2.0 * t))
    assert f_dom == pytest.approx(2.0, abs=0.3) and 0 <= ent <= 1
    assert cm.spectral_features(np.full(40, 3.0)) == (pytest.approx(np.nan, nan_ok=True), pytest.approx(np.nan, nan_ok=True))


def test_sample_entropy_regular_below_random():
    rng = np.random.default_rng(0)
    assert cm.sample_entropy(np.sin(np.arange(200) * 0.3)) < cm.sample_entropy(rng.normal(size=200))
    assert np.isnan(cm.sample_entropy(np.zeros(50)))
    assert np.isnan(cm.sample_entropy(np.arange(4.0)))


def test_attempt_repeatability_and_normalization():
    traj = pd.DataFrame({"nfl_id": [1, 1, 1, 2], "drill_name": ["D"] * 4, "attempt": [1, 2, 3, 1], "draft_year": 2024,
                         "max_speed": [5.0, 6.0, 5.5, 7.0], "path_length": [10.0, 11.0, 12.0, 9.0], "duration": [4.0, 4.2, 4.1, 4.0]})
    out = cm.attempt_repeatability(traj, cols=["max_speed"])
    p1 = out[out["nfl_id"] == 1]
    assert np.isnan(p1["max_speed_delta_prev"].iloc[0]) and p1["max_speed_delta_prev"].iloc[1] == pytest.approx(1.0)
    assert p1["max_speed_minus_player_mean"].sum() == pytest.approx(0.0)
    assert np.isnan(out[out["nfl_id"] == 2]["max_speed_delta_prev"].iloc[0])
    res = pd.DataFrame({"nfl_id": [1, 2], "combine_height": [72.0, 78.0], "combine_weight": [200.0, 300.0],
                        "wing_span": [74.0, 80.0], "combine_position": ["DB", "DB"]})
    norm = cm.add_normalized(out, res)
    h = 72 * 0.0254
    assert norm.loc[norm["nfl_id"] == 1, "froude_max"].iloc[0] == pytest.approx((5.0 * 0.9144) ** 2 / (9.81 * h))
    assert "max_speed_z" in norm
