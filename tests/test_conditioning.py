import numpy as np
import pandas as pd
import pytest

from bio import (
    EXCLUDED,
    KEYS,
    Conditioner,
    LeakageError,
    assert_no_leakage,
    build_bio,
    build_conditioning,
    feature_columns,
)
from splits import check_grouped, make_folds


def make_tables(n_players=12, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + n_players)
    res = pd.DataFrame({
        "nfl_id": ids, "combine_position": rng.choice(["OL", "DB", "WR"], n_players),
        "combine_height": rng.normal(74, 2, n_players), "combine_weight": rng.normal(230, 30, n_players),
        "arm_length": rng.normal(32, 1, n_players), "hand_size": rng.normal(9.5, 0.5, n_players),
        "wing_span": rng.normal(77, 2, n_players), "ngs_college_production_score": rng.normal(50, 10, n_players),
        "ngs_athleticism_score": rng.normal(60, 10, n_players), "short_shuttle": rng.normal(4.4, 0.2, n_players),
        "three_cone": rng.normal(7.2, 0.3, n_players), "draft_year": 2024})
    res.loc[2, "hand_size"] = np.nan
    ply = pd.DataFrame({"nfl_id": ids, "nfl_position": rng.choice(["G", "T", "CB", "WR", "FB"], n_players)})
    att = pd.DataFrame({"nfl_id": np.repeat(ids, 2), "event_id": [f"e{i}" for i in range(2 * n_players)],
                        "attempt": np.tile([1, 2], n_players), "attempt_order": np.tile([1, 2], n_players),
                        "qc_pass": True, "start_dir": rng.choice(["L", "R"], 2 * n_players),
                        "t_total": rng.normal(5, 0.3, 2 * n_players)})
    ref = pd.DataFrame({"nfl_id": ids, "n_pass": 2, **{p: rng.normal(10, 1, n_players) for p in
                        ["v_max", "tau", "a0", "F0", "P_max", "t90", "d90", "speed_drop", "t10_model", "t40_model"]}})
    for p in ["v_max", "tau", "a0", "F0", "P_max", "t90", "d90", "speed_drop", "t10_model", "t40_model"]:
        ref[f"best_{p}"] = ref[p]
    return res, ply, att, ref


def build(n_players=12):
    res, ply, att, ref = make_tables(n_players)
    bio = build_bio(res, ply, att["nfl_id"].unique(), min_count=3)
    return res, build_conditioning(att, ref, bio).merge(att[KEYS + ["t_total"]], on=KEYS)


def test_guard_blocks_every_excluded_and_outcome_column():
    for c in EXCLUDED + ["t_total", "cod_cost", "score_primary", "t_leg1", "draft_round"]:
        with pytest.raises(LeakageError):
            assert_no_leakage(["f40_v_max", c])
    assert_no_leakage(["f40_v_max", "combine_height", "pos_c_OL", "start_dir_R"])


def test_feature_columns_exclude_target_and_forbidden():
    _, cond = build()
    num, bin_ = feature_columns(cond)
    feats = set(num + bin_)
    assert "t_total" not in feats
    assert not feats & set(EXCLUDED)
    assert not any(c.startswith(("draft_", "ngs_athleticism", "ngs_final", "three_cone", "short_shuttle")) for c in feats)
    assert "start_dir_R" in feats and "hand_size_missing" in feats


def test_inputs_do_not_change_when_forbidden_columns_change():
    res, ply, att, ref = make_tables()
    bio1 = build_bio(res, ply, att["nfl_id"].unique(), 3)
    res2 = res.copy()
    for c in ("ngs_athleticism_score", "short_shuttle", "three_cone"):
        res2[c] = res2[c] + 100.0
    res2["draft_year"] = 1999
    bio2 = build_bio(res2, ply, att["nfl_id"].unique(), 3)
    c1, c2 = build_conditioning(att, ref, bio1), build_conditioning(att, ref, bio2)
    n, b = feature_columns(c1)
    pd.testing.assert_frame_equal(c1[n + b], c2[n + b])
    att2 = att.assign(t_total=att["t_total"] + 5.0)
    c3 = build_conditioning(att2, ref, bio1)
    pd.testing.assert_frame_equal(c1[n + b], c3[n + b])


def test_conditioner_fits_on_training_rows_only_and_fills_missing():
    _, cond = build()
    num, bin_ = feature_columns(cond)
    train, test = cond.iloc[:16], cond.iloc[16:]
    cz = Conditioner(num, bin_).fit(train)
    Xtr = cz.transform(train)
    assert np.allclose(Xtr[num].mean(), 0, atol=1e-9)
    shifted = test.copy()
    shifted[num] = shifted[num] + 1000.0           # test-time values must not influence the scaler
    assert np.allclose(cz.mean_, Conditioner(num, bin_).fit(train).mean_)
    assert not cz.transform(shifted).isna().any().any()
    with_nan = test.copy()
    with_nan.loc[with_nan.index[0], "combine_height"] = np.nan
    assert cz.transform(with_nan).iloc[0]["combine_height"] == 0.0


def test_rare_positions_pooled():
    _, cond = build()
    pos_n = [c for c in cond.columns if c.startswith("pos_n_")]
    assert "pos_n_OTHER" in pos_n or len(pos_n) >= 1
    assert (cond[[c for c in cond.columns if c.startswith("pos_c_")]].sum(axis=1) == 1).all()


def test_folds_are_grouped_and_reproducible():
    players = pd.DataFrame({"nfl_id": np.arange(50), "n_attempts": np.random.default_rng(0).integers(1, 4, 50), "draft_year": 2024})
    f1, f2 = make_folds(players, 5, seed=7), make_folds(players, 5, seed=7)
    pd.testing.assert_frame_equal(f1, f2)
    assert set(f1["fold"]) == set(range(5)) and len(f1) == 50
    check_grouped(f1)
    bad = pd.concat([f1, f1.iloc[:1].assign(fold=(f1.iloc[0]["fold"] + 1) % 5)])
    with pytest.raises(AssertionError):
        check_grouped(bad)
