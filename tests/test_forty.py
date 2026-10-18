import numpy as np
import pandas as pd
import pytest

from config import pipeline_cfg
from forty import analyze_attempt, build_forty, t_ref, x_ref


def synth_attempt(v_max, tau, t0, n=55, noise=0.0, seed=0, start_x=10.0, direction=1.0, event="e0", nfl_id=1):
    """10 Hz sprint; the first frame is 0.1 s after the athlete's own start offset (t0 > 0 means late start)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.1
    s = v_max * (1 - np.exp(-np.clip(t - t0, 0, None) / tau)) + rng.normal(0, noise, n)
    x = start_x + direction * x_ref(t - t0, v_max, tau)
    return pd.DataFrame({
        "event_id": event, "nfl_id": nfl_id, "attempt": 1, "draft_year": 2024,
        "time": pd.Timestamp("2024-03-01 12:00:00") + pd.to_timedelta(t, unit="s"),
        "x": x, "y": 26.0 + 0.0 * t, "s": s})


def test_t_ref_inverts_x_ref():
    for v_max, tau in [(10.0, 1.2), (12.5, 1.6), (8.8, 0.9)]:
        d = np.array([0.5, 5.0, 10.0, 40.0])
        assert np.allclose(x_ref(t_ref(d, v_max, tau), v_max, tau), d, atol=1e-6)


@pytest.mark.parametrize("v_max,tau,t0", [(10.5, 1.3, 0.0), (11.8, 1.1, 0.15), (9.4, 1.6, 0.05)])
def test_fit_recovers_known_parameters(v_max, tau, t0):
    cfg = pipeline_cfg()["forty"]
    out = analyze_attempt(synth_attempt(v_max, tau, t0), mass_kg=100.0, cfg=cfg)
    assert out["v_max"] == pytest.approx(v_max, rel=0.01)
    assert out["tau"] == pytest.approx(tau, rel=0.02)
    assert out["t0"] == pytest.approx(t0, abs=0.03)
    assert out["fit_rmse"] < 1e-3
    assert out["t40_model"] == pytest.approx(float(t_ref(40.0, v_max, tau)), rel=0.01)
    assert out["a0"] == pytest.approx(v_max / tau, rel=0.02)


def test_fit_robust_to_noise_and_direction():
    cfg = pipeline_cfg()["forty"]
    f = synth_attempt(11.0, 1.25, 0.1, noise=0.1, direction=-1.0, start_x=110.0)
    out = analyze_attempt(f, mass_kg=100.0, cfg=cfg)
    assert out["v_max"] == pytest.approx(11.0, rel=0.03)
    assert out["tau"] == pytest.approx(1.25, rel=0.1)
    assert out["reached_40"]
    assert out["p_span"] > 39.0


def test_power_and_force_units():
    cfg = pipeline_cfg()["forty"]
    out = analyze_attempt(synth_attempt(10.0, 1.25, 0.0), mass_kg=100.0, cfg=cfg)
    a0_mps2 = 10.0 / 1.25 * 0.9144
    assert out["F0"] == pytest.approx(100.0 * a0_mps2, rel=0.02)
    assert out["P_max"] == pytest.approx(out["F0"] * 10.0 * 0.9144 / 4, rel=0.03)


def test_qc_flags_bad_fit_and_builds_reference():
    cfg = pipeline_cfg()["forty"]
    good = [synth_attempt(10.5 + 0.1 * i, 1.3, 0.05, noise=0.05, seed=i, event=f"g{i}", nfl_id=1 + i % 2) for i in range(6)]
    bad = synth_attempt(10.0, 1.3, 0.0, event="bad", nfl_id=1)
    rng = np.random.default_rng(3)
    bad["s"] = rng.uniform(0, 11, len(bad))  # garbage speed trace
    trk = pd.concat(good + [bad], ignore_index=True)
    # official times consistent with the model plus the usual clock offset
    res = pd.DataFrame({"nfl_id": [1, 2], "combine_weight": [220.0, 250.0],
                        "forty": [4.7, 4.7], "ten_yd_split": [1.62, 1.62]})
    att, ref = build_forty(trk, res, cfg)
    assert not att.loc[att.event_id == "bad", "qc_pass"].iloc[0]
    assert "high_rmse" in att.loc[att.event_id == "bad", "qc_reason"].iloc[0]
    assert set(ref["nfl_id"]) == {1, 2}
    assert (ref["n_pass"] <= ref["n_attempts"]).all()
    assert (ref["best_t40_model"] <= ref["t40_model"] + 1e-9).all()
