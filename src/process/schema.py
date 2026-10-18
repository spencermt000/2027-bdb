"""Output convention for every experiment: outputs/<exp_id>/attempt_scores.parquet, one row per attempt."""
import numpy as np
import pandas as pd

from config import paths

EXP_IDS = {"baseline", "T1", "T2", "T3", "T4", "T5"}
KEYS = ["nfl_id", "event_id", "attempt"]
REQUIRED = KEYS + ["exp_id", "score_primary"]


class SchemaError(ValueError):
    pass


def validate_scores(df: pd.DataFrame, exp_id: str) -> None:
    """Raise SchemaError if `df` violates the output convention."""
    if exp_id not in EXP_IDS:
        raise SchemaError(f"unknown exp_id {exp_id!r}; expected one of {sorted(EXP_IDS)}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if not pd.api.types.is_integer_dtype(df["nfl_id"]) or not pd.api.types.is_integer_dtype(df["attempt"]):
        raise SchemaError("nfl_id and attempt must be integers")
    if not df["event_id"].map(lambda x: isinstance(x, str)).all():
        raise SchemaError("event_id must be str")
    if not (df["exp_id"] == exp_id).all():
        raise SchemaError(f"exp_id column must equal {exp_id!r} in every row")
    if df.duplicated(KEYS).any():
        raise SchemaError("duplicate (nfl_id, event_id, attempt) rows")
    if len(df) == 0:
        raise SchemaError("no rows")
    extras = [c for c in df.columns if c.startswith("score_")]
    for c in extras:
        if not pd.api.types.is_float_dtype(df[c]):
            raise SchemaError(f"{c} must be float, got {df[c].dtype}")
    if not np.isfinite(df["score_primary"]).all():
        raise SchemaError("score_primary must be finite in every row")
    unknown = [c for c in df.columns if c not in REQUIRED and not c.startswith("score_")]
    if unknown:
        raise SchemaError(f"unexpected columns (extras must start with 'score_'): {unknown}")


def write_scores(df: pd.DataFrame, exp_id: str, outputs_dir=None):
    """Validate and write outputs/<exp_id>/attempt_scores.parquet. Every experiment calls this at the end."""
    validate_scores(df, exp_id)
    out = (outputs_dir or paths()["outputs"]) / exp_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "attempt_scores.parquet"
    df.to_parquet(path, index=False)
    return path
