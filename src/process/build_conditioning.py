"""Stage C driver: bio table, raw conditioning table and the shared folds.

Needs build_forty.py and build_shuttle.py first.
Outputs: data/derived/bio.parquet, data/derived/conditioning.parquet, data/derived/folds.parquet
Usage:   python src/process/build_conditioning.py
"""
import pandas as pd

from bio import build_bio, build_conditioning, feature_columns
from config import paths, pipeline_cfg, seed_everything
from loaders import load_players, load_results
from splits import check_grouped, make_folds


def main() -> None:
    cfg = pipeline_cfg()
    seed_everything(cfg["seed"])
    p = paths()
    att = pd.read_parquet(p["derived"] / "shuttle_attempt_features.parquet")
    ref = pd.read_parquet(p["derived"] / "forty_reference.parquet")

    bio = build_bio(load_results(), load_players(), att["nfl_id"].unique(), cfg["eval"]["rare_category_min"])
    cond = build_conditioning(att, ref, bio, cfg["shuttle"]["forty_reference"])
    cond = cond.merge(att[["nfl_id", "event_id", "attempt", "t_total"]], on=["nfl_id", "event_id", "attempt"])

    players = att.groupby("nfl_id").agg(n_attempts=("event_id", "size"), draft_year=("draft_year", "first")).reset_index()
    folds = make_folds(players, cfg["eval"]["n_folds"], cfg["seed"])
    check_grouped(folds)

    bio.to_parquet(p["derived"] / "bio.parquet", index=False)
    cond.to_parquet(p["derived"] / "conditioning.parquet", index=False)
    folds.to_parquet(p["derived"] / "folds.parquet", index=False)
    numeric, binary = feature_columns(cond)
    print(f"conditioning: {len(cond)} attempts ({int(cond['eval_ok'].sum())} eval_ok), {len(numeric)} numeric + {len(binary)} binary columns")
    print(f"folds: {folds['fold'].value_counts().sort_index().to_dict()} players per fold")


if __name__ == "__main__":
    main()
