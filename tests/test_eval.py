import numpy as np
import pandas as pd
import pytest

from baseline import oof_predictions, quantile_position, score_attempts
from bio import KEYS
from harness import agreement, build_report, icc1, incremental_validity, reliability
from schema import SchemaError, validate_scores, write_scores
from splits import make_folds


def good_scores(exp="T1", n=6):
    return pd.DataFrame({"nfl_id": np.arange(n, dtype="int64"), "event_id": [f"e{i}" for i in range(n)],
                         "attempt": np.ones(n, dtype="int64"), "exp_id": exp, "score_primary": np.linspace(-1, 1, n),
                         "score_extra": np.zeros(n), "score_uncertainty": np.nan})


def test_schema_accepts_good_and_rejects_bad():
    validate_scores(good_scores(), "T1")
    with pytest.raises(SchemaError):
        validate_scores(good_scores(), "T2")                                    # exp_id mismatch
    with pytest.raises(SchemaError):
        validate_scores(good_scores().drop(columns="score_primary"), "T1")
    with pytest.raises(SchemaError):
        validate_scores(good_scores().assign(score_primary=np.nan), "T1")
    with pytest.raises(SchemaError):
        validate_scores(pd.concat([good_scores(), good_scores().iloc[:1]]), "T1")  # duplicate key
    with pytest.raises(SchemaError):
        validate_scores(good_scores().assign(extra_col=1), "T1")                 # non score_ extra
    with pytest.raises(SchemaError):
        validate_scores(good_scores().assign(score_x=1), "T1")                   # int extra, not float
    with pytest.raises(SchemaError):
        validate_scores(good_scores().assign(event_id=1), "T1")


def test_write_scores_validates_first(tmp_path):
    with pytest.raises(SchemaError):
        write_scores(good_scores().assign(score_primary=np.inf), "T1", outputs_dir=tmp_path)
    assert not (tmp_path / "T1").exists()
    path = write_scores(good_scores(), "T1", outputs_dir=tmp_path)
    assert pd.read_parquet(path).shape[0] == 6


def test_icc_known_values():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(300), 2)
    signal = np.repeat(rng.normal(0, 1, 300), 2)
    hi = pd.Series(signal + rng.normal(0, 0.3, 600))     # true ICC ~ 0.92
    lo = pd.Series(signal * 0.2 + rng.normal(0, 1, 600))  # true ICC ~ 0.04
    assert icc1(hi, pd.Series(g)) == pytest.approx(0.92, abs=0.04)
    assert icc1(lo, pd.Series(g)) < 0.15
    assert np.isnan(icc1(pd.Series([1.0, 2.0]), pd.Series([1, 2])))           # singletons only


def test_reliability_uses_first_two_attempts():
    rng = np.random.default_rng(1)
    rows = []
    for p in range(40):
        s = rng.normal()
        for k in (1, 2):
            rows.append({"nfl_id": p, "event_id": f"{p}-{k}", "attempt_order": k, "score_primary": s + rng.normal(0, 0.2)})
    r = reliability(pd.DataFrame(rows), n_boot=100)
    assert r["n_players"] == 40 and r["icc"] > 0.8 and r["spearman_12"] > 0.8
    assert r["icc_ci"][0] <= r["icc"] <= r["icc_ci"][1]


def test_partial_correlation_removes_control_effect():
    rng = np.random.default_rng(2)
    n = 300
    ctrl = rng.normal(size=n)
    score = ctrl + rng.normal(0, 0.5, n)
    outcome = 2 * ctrl + rng.normal(0, 0.5, n)                  # score only informative through the control
    d = pd.DataFrame({"score": score, "o": outcome, "c": ctrl})
    v = incremental_validity(d, "o", ["c"])
    assert abs(v["partial_r"]) < 0.15 and v["r2_gain"] < 0.02
    d["o2"] = outcome + 1.0 * (score - ctrl)                     # genuine extra information
    v2 = incremental_validity(d, "o2", ["c"])
    assert v2["partial_r"] > 0.4 and v2["r2_gain"] > 0.05


def test_quantile_position_monotone_and_calibrated():
    q = np.tile([4.9, 5.0, 5.2], (5, 1))
    u = quantile_position(np.array([4.9, 5.0, 5.2, 4.0, 6.5]), q)
    assert u[0] == pytest.approx(0.1) and u[1] == pytest.approx(0.5) and u[2] == pytest.approx(0.9)
    assert u[3] == 0.01 and u[4] == 0.99                              # clipped tails
    y = np.linspace(4.5, 5.6, 30)
    assert (np.diff(quantile_position(y, np.tile([4.9, 5.0, 5.2], (30, 1)))) >= 0).all()


def test_agreement_matrix_and_disagreement_list():
    a = pd.Series(np.arange(20.0), index=np.arange(20))
    b = a.copy()
    b.iloc[3], b.iloc[16] = 19.0, 0.0                              # players 3 and 16 flip ranks
    mat, top = agreement({"A": a, "B": b})
    assert mat.loc["A", "A"] == pytest.approx(1.0) and 0.5 < mat.loc["A", "B"] < 1.0
    assert set(top.index[:2]) == {3, 16}


def test_baseline_oof_has_no_player_overlap_and_scores_validate():
    rng = np.random.default_rng(0)
    n = 80
    ids = np.repeat(np.arange(40), 2)
    x = rng.normal(size=n)
    data = pd.DataFrame({"nfl_id": ids, "event_id": [f"e{i}" for i in range(n)], "attempt": np.tile([1, 2], 40),
                         "x1": x, "x2": rng.normal(size=n), "ind": rng.integers(0, 2, n).astype(float),
                         "t_total": 5 + 0.3 * x + rng.normal(0, 0.1, n), "draft_year": 2024})
    folds = make_folds(pd.DataFrame({"nfl_id": np.arange(40), "n_attempts": 2, "draft_year": 2024}), 5, seed=1)
    cfg = {"quantiles": [0.1, 0.5, 0.9], "lgbm": {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 4, "min_child_samples": 4}}
    q = oof_predictions(data, folds, ["x1", "x2"], ["ind"], cfg, seed=0, mode="kfold")
    assert not q.isna().any().any() and (q["q10"] <= q["q50"]).all() and (q["q50"] <= q["q90"]).all()
    out = score_attempts(data, q)
    validate_scores(out, "baseline")
    assert np.corrcoef(out["score_resid_median"], 5 + 0.3 * x - data["t_total"])[0, 1] > 0.3


def test_report_runs_on_baseline_alone_and_with_two_experiments(tmp_path):
    rng = np.random.default_rng(3)
    n_p = 30
    ctx = pd.DataFrame({"nfl_id": np.repeat(np.arange(n_p), 2), "event_id": [f"e{i}" for i in range(2 * n_p)],
                        "attempt": np.tile([1, 2], n_p), "attempt_order": np.tile([1, 2], n_p),
                        "t_total": rng.normal(5, 0.3, 2 * n_p), "f40_v_max": np.repeat(rng.normal(10, 1, n_p), 2),
                        "f40_tau": np.repeat(rng.normal(1.3, 0.1, n_p), 2)})
    res = pd.DataFrame({"nfl_id": np.arange(n_p), "ngs_athleticism_score": rng.normal(60, 10, n_p), "three_cone": rng.normal(7.2, 0.3, n_p)})

    def scores(exp):
        return ctx[KEYS].assign(exp_id=exp, score_primary=rng.normal(size=len(ctx)))
    cfg = {"n_boot": 50, "validity_outcomes": ["three_cone"]}
    one = build_report({"baseline": scores("baseline")}, ctx, res, cfg, seed=0)
    assert "Reliability" in one and "Only one experiment present" in one
    two = build_report({"baseline": scores("baseline"), "T1": scores("T1")}, ctx, res, cfg, seed=0)
    assert "largest disagreement" in two
    assert "No `attempt_scores.parquet`" in build_report({}, ctx, res, cfg, seed=0)
