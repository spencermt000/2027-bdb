import numpy as np
import pandas as pd
import pytest

from config import pipeline_cfg
from shuttle import analyze_attempt, build_shuttle, find_turns
from smoothing import rts_smooth_1d, smooth_xy

CFG = pipeline_cfg()["shuttle"]
REF = {"v_max": 10.5, "tau": 1.3, "a0": 10.5 / 1.3}
T_TOTAL, AMP, LEAD = 4.4, 4.0, 0.5
EXPONENT = 1.3


def synth_shuttle(direction=1.0, angle=0.0, lead=LEAD, noise=0.0, seed=0, n_tail=5, event="e0", nfl_id=1, truncate=None):
    """A 5-10-5-like trace: p = A sin(2 pi w(u)), w = u**1.3, so plants sit at w = 1/4 and 3/4 and the finish at u = 1.

    The athlete stands still for `lead` seconds first. Mirrored by `direction`, rotated by `angle` (rad) in the plane.
    """
    t = np.arange(0.0, lead + T_TOTAL + n_tail * 0.1, 0.1)
    u = np.clip((t - lead) / T_TOTAL, 0, None)
    p = direction * AMP * np.sin(2 * np.pi * u**EXPONENT)
    rng = np.random.default_rng(seed)
    x, y = 60 + p * np.cos(angle), 26 + p * np.sin(angle)
    x, y = x + rng.normal(0, noise, len(t)), y + rng.normal(0, noise, len(t))
    df = pd.DataFrame({"event_id": event, "nfl_id": nfl_id, "attempt": 1, "draft_year": 2024,
                       "time": pd.Timestamp("2024-03-01 12:00:00") + pd.to_timedelta(t, unit="s"), "x": x, "y": y})
    return df.iloc[:truncate] if truncate else df


def true_plant_times():
    return [LEAD + T_TOTAL * (w ** (1 / EXPONENT)) for w in (0.25, 0.75)]


def test_rts_recovers_constant_acceleration_and_reports_variance():
    t = np.arange(0, 3, 0.1)
    z = 1.0 * t + 0.5 * 2.0 * t**2
    z_noisy = z + np.random.default_rng(1).normal(0, 0.05, len(t))
    x, P = rts_smooth_1d(z_noisy, t, sigma=0.05, q=1.0)
    assert np.abs(x[2:-2, 0] - z[2:-2]).mean() < 0.05
    assert np.abs(x[3:-3, 1] - (1.0 + 2.0 * t)[3:-3]).mean() < 0.3
    assert np.all(P[:, 0, 0] > 0) and P[:, 0, 0].max() < 0.05**2 * 2
    assert P[len(t) // 2, 0, 0] < P[0, 0, 0]  # smoothed interior is tighter than the edge


def test_smooth_xy_projection():
    t = np.arange(0, 2, 0.1)
    xy = np.c_[10 + 3 * t, 5 + 4 * t]                 # speed 5 along (0.6, 0.8)
    out = smooth_xy(xy, t, np.array([0.6, 0.8]), sigma=0.01, q=0.1)
    assert np.allclose(out["v"][5:-5], 5.0, atol=0.1)
    assert np.allclose(out["speed"][5:-5], 5.0, atol=0.1)


def test_find_turns_prominence_ignores_wiggles():
    t = np.arange(0, 5, 0.1)
    p = 4 * np.sin(2 * np.pi * t / 4.4) + 0.2 * np.sin(20 * t)
    v = np.gradient(p, t)
    times, pos, _, n = find_turns(t, p, v, prominence=2.0)
    assert n == 2 and len(times) == 2
    assert times[0] == pytest.approx(1.1, abs=0.1) and times[1] == pytest.approx(3.3, abs=0.1)


def test_plants_phases_and_features_on_known_shuttle():
    row, fr = analyze_attempt(synth_shuttle(), mass_kg=100.0, ref=REF, cfg=CFG)
    t1, t2 = true_plant_times()
    assert row["n_plants_found"] == 2
    assert row["t_plant_1"] == pytest.approx(t1, abs=0.08) and row["t_plant_2"] == pytest.approx(t2, abs=0.08)
    assert row["p_plant_1"] - row["p_line"] == pytest.approx(AMP, abs=0.15)
    assert row["p_plant_2"] - row["p_line"] == pytest.approx(-AMP, abs=0.15)
    assert row["start_dir"] == "R"
    assert LEAD - 0.3 < row["t_start"] < LEAD + 0.1               # idle lead is not counted
    assert row["t_total"] == pytest.approx(T_TOTAL, abs=0.35)
    assert row["leg_dist_1"] == pytest.approx(AMP, abs=0.2) and row["leg_dist_2"] == pytest.approx(2 * AMP, abs=0.2)
    assert row["entry_speed_2"] > row["entry_speed_1"] > 0
    assert row["brake_onset_dist_2"] > 0 and row["peak_decel_2"] >= row["mean_decel_2"] > 0
    assert row["cod_cost"] == pytest.approx(row["cod_cost_plant_1"] + row["cod_cost_plant_2"])
    assert row["cod_cost"] == pytest.approx(row["t_total"] - row["t_ref_total"])
    assert row["qc_plants_ok"]
    # frame table: phases in order, each present, plants contain their turn, d_line positive on approach
    assert list(np.unique(fr["phase_id"])) == [0, 1, 2, 3, 4]
    assert (np.diff(fr["phase_id"]) >= 0).all()
    for k, tt in ((1, t1), (3, t2)):
        win = fr[fr["phase_id"] == k]
        assert win["t"].min() < tt < win["t"].max()
    first = fr.iloc[0]
    assert first["d_line"] == pytest.approx(AMP, abs=0.2)
    assert fr["d_line"].iloc[-1] < 1.0


def test_mirror_and_rotation_invariance():
    base, _ = analyze_attempt(synth_shuttle(), 100.0, REF, CFG)
    mirror, _ = analyze_attempt(synth_shuttle(direction=-1.0), 100.0, REF, CFG)
    rot, _ = analyze_attempt(synth_shuttle(angle=0.5), 100.0, REF, CFG)
    assert mirror["start_dir"] == "L" and base["start_dir"] == "R"
    for k in ("t_total", "entry_speed_1", "entry_speed_2", "brake_dist_2", "cod_cost", "dwell_1"):
        assert mirror[k] == pytest.approx(base[k], abs=0.05)
        assert rot[k] == pytest.approx(base[k], abs=0.05)


def test_noise_robustness():
    row, _ = analyze_attempt(synth_shuttle(noise=0.04), 100.0, REF, CFG)
    t1, t2 = true_plant_times()
    assert row["n_plants_found"] == 2
    assert row["t_plant_1"] == pytest.approx(t1, abs=0.15) and row["t_plant_2"] == pytest.approx(t2, abs=0.15)


def test_qc_flags_missing_plant_and_short_clips():
    row, fr = analyze_attempt(synth_shuttle(truncate=30), 100.0, REF, CFG)       # ends before plant 2
    assert not row["qc_pass"] and "n_plants=1" in row["qc_reason"] and fr is None
    row, fr = analyze_attempt(synth_shuttle(truncate=10), 100.0, REF, CFG)
    assert "too_few_frames" in row["qc_reason"]
    # clip ends before the finish -> no t_total, plants still fine
    # truncate at 4.8 s: past plant 2 (4.0 s + lead), before the finish at 4.9 s
    row, _ = analyze_attempt(synth_shuttle(n_tail=0, truncate=48), 100.0, REF, CFG)
    assert "no_finish" in row["qc_reason"] and row["qc_plants_ok"] and np.isnan(row["t_total"])


def test_no_forty_reference_still_gives_plant_features():
    row, _ = analyze_attempt(synth_shuttle(), 100.0, None, CFG)
    assert "no_forty_ref" in row["qc_reason"]
    assert np.isnan(row["entry_ratio_1"]) and np.isfinite(row["entry_speed_1"]) and "cod_cost" not in row


def test_build_shuttle_attempt_order_and_tables():
    tracks = pd.concat([synth_shuttle(event="a", nfl_id=1, seed=1, noise=0.02),
                        synth_shuttle(event="b", nfl_id=1, seed=2, noise=0.02),
                        synth_shuttle(event="c", nfl_id=2, direction=-1.0, seed=3, noise=0.02)], ignore_index=True)
    tracks.loc[tracks.event_id == "b", "time"] += pd.Timedelta(hours=1)
    res = pd.DataFrame({"nfl_id": [1, 2], "combine_weight": [200.0, 220.0], "short_shuttle": [4.4, 4.5]})
    ref = pd.DataFrame({"nfl_id": [1, 2], "n_pass": [2, 1], **{k: [v, v] for k, v in REF.items()},
                        **{f"best_{k}": [v, v] for k, v in REF.items()}})
    att, fr = build_shuttle(tracks, res, ref, CFG)
    assert att.set_index("event_id").loc[["a", "b"], "attempt_order"].tolist() == [1, 2]
    assert set(fr["event_id"]) == {"a", "b", "c"}
    assert {"p", "v", "a", "p_var", "d_line", "phase_id", "t_since_start", "frame_idx"} <= set(fr.columns)
    assert att["qc_plants_ok"].all()
