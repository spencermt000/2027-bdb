"""Stage C: bio table, conditioning vector, and the leakage guard (spec 1.0).

Model inputs are the forty reference, the bio table and the attempt's start direction. Everything in EXCLUDED must
never reach a model: the NGS athleticism / final scores (likely include shuttle results; leakage check only), the
official shuttle time (target-adjacent), three_cone (out of scope) and all draft variables.
"""
import numpy as np
import pandas as pd

BIO_NUMERIC = ["combine_height", "combine_weight", "arm_length", "hand_size", "wing_span", "ngs_college_production_score"]
BIO_CATEGORICAL = ["combine_position", "nfl_position"]
FORTY_PARAMS = ["v_max", "tau", "a0", "F0", "P_max", "t90", "d90", "speed_drop", "t10_model", "t40_model"]

EXCLUDED = ["ngs_athleticism_score", "ngs_final_score", "short_shuttle", "three_cone",
            "draft_round", "draft_pick_within_round", "draft_overall_pick", "draft_year"]
EXCLUDED_PREFIXES = ("draft_", "ngs_athleticism", "ngs_final", "short_shuttle", "three_cone")
# attempt outcomes derived from the shuttle itself: targets or target-adjacent, never conditioning inputs
OUTCOME_COLUMNS = ["t_total", "cod_cost", "attempt_order"]
OUTCOME_PREFIXES = ("score_", "t_leg", "cod_cost", "resid_")
KEYS = ["nfl_id", "event_id", "attempt"]


class LeakageError(ValueError):
    pass


def assert_no_leakage(columns) -> None:
    """Raise if any model-input column is an excluded quantity."""
    bad = [c for c in columns if c in EXCLUDED or c in OUTCOME_COLUMNS or c.startswith(EXCLUDED_PREFIXES + OUTCOME_PREFIXES)]
    if bad:
        raise LeakageError(f"excluded columns reached a model input: {bad}")


def _pool_rare(s: pd.Series, min_count: int) -> pd.Series:
    counts = s.value_counts()
    return s.where(s.map(counts) >= min_count, "OTHER")


def build_bio(results: pd.DataFrame, players: pd.DataFrame, shuttle_players, min_count: int) -> pd.DataFrame:
    """One row per player: bio numerics, missingness indicators, position one-hots.

    Missing values are kept (imputed later from training folds only) and flagged: opt-outs are not random.
    Missingness indicators exist for every numeric column with any null in the full results table; this depends on
    the data's missingness only, not on any target. Rare position levels are pooled using shuttle-player counts.
    """
    ids = pd.Index(sorted(set(shuttle_players)), name="nfl_id")
    res = results.set_index("nfl_id").reindex(ids)
    ply = players.set_index("nfl_id")[["nfl_position"]].reindex(ids)
    bio = res[["combine_position", *BIO_NUMERIC]].join(ply)
    for c in BIO_NUMERIC:
        if results[c].isna().any():
            bio[f"{c}_missing"] = bio[c].isna().astype(float)
    for c, tag in (("combine_position", "pos_c"), ("nfl_position", "pos_n")):
        lvl = _pool_rare(bio[c].fillna("UNKNOWN"), min_count)
        bio = bio.join(pd.get_dummies(lvl, prefix=tag, dtype=float))
    return bio.drop(columns=BIO_CATEGORICAL).reset_index()


def build_conditioning(att: pd.DataFrame, forty_ref: pd.DataFrame, bio: pd.DataFrame, reference: str = "mean") -> pd.DataFrame:
    """Raw (unstandardized) conditioning table, one row per shuttle attempt.

    Columns: KEYS, `eval_ok` (QC pass), forty reference as f40_*, `f40_missing`, bio columns, `start_dir_R`.
    `reference` selects the player-level mean or best-attempt forty parameters.
    """
    prefix = "best_" if reference == "best" else ""
    f40 = forty_ref[forty_ref["n_pass"] > 0].set_index("nfl_id")[[prefix + p for p in FORTY_PARAMS]]
    f40.columns = [f"f40_{p}" for p in FORTY_PARAMS]
    out = att[KEYS + ["attempt_order", "qc_pass", "start_dir"]].rename(columns={"qc_pass": "eval_ok"})
    out = out.join(f40, on="nfl_id").join(bio.set_index("nfl_id"), on="nfl_id")
    out["f40_missing"] = out["f40_t40_model"].isna().astype(float)
    out["start_dir_R"] = (out["start_dir"] == "R").astype(float).where(out["start_dir"].notna())
    return out.drop(columns="start_dir")


def feature_columns(cond: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numeric columns to standardize, binary/indicator columns left as is).

    An explicit whitelist: forty reference, bio numerics, missingness indicators, position one-hots, start direction.
    Anything else in the table (keys, targets, bookkeeping) is never a model input.
    """
    numeric = [f"f40_{p}" for p in FORTY_PARAMS] + BIO_NUMERIC
    binary = (["f40_missing", "start_dir_R"]
              + [c for c in cond.columns if c.endswith("_missing") and c != "f40_missing"]
              + [c for c in cond.columns if c.startswith(("pos_c_", "pos_n_"))])
    absent = [c for c in numeric + binary if c not in cond.columns]
    if absent:
        raise KeyError(f"conditioning table is missing columns: {absent}")
    assert_no_leakage(numeric + binary)
    return numeric, binary


class Conditioner:
    """Standardizes the conditioning vector. Fit on training folds only; missing values -> training mean (0 after scaling)."""

    def __init__(self, numeric: list[str], binary: list[str]):
        assert_no_leakage(numeric + binary)
        self.numeric, self.binary = numeric, binary

    def fit(self, train: pd.DataFrame) -> "Conditioner":
        self.mean_ = train[self.numeric].mean()
        self.std_ = train[self.numeric].std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.bin_fill_ = train[self.binary].mean().fillna(0.0).round()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num = ((df[self.numeric] - self.mean_) / self.std_).fillna(0.0)
        bin_ = df[self.binary].fillna(self.bin_fill_)
        out = pd.concat([num, bin_], axis=1)
        assert not np.isnan(out.to_numpy()).any()
        return out

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.binary
