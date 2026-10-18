"""Standardize game tracking so every play reads "offense on the left, driving right".

After standardization, the offense always moves toward +x (the defense's end zone is at
x=120), regardless of the raw `play_direction`. Also adds the line of scrimmage in
standardized x, so positions can be expressed relative to the ball.

Inputs (raw/):  game_tracking_{season}.csv, player_play.csv
Output:         processed/game_tracking_std_{season}.parquet
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw"
OUT = ROOT / "processed"

FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
SEASONS = (2023, 2024, 2025)


def play_context(player_play: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, play_id): direction and standardized line of scrimmage."""
    cols = ["game_id", "play_id", "possession_team", "yardline_side",
            "yardline_number", "play_direction"]
    ctx = player_play[cols].drop_duplicates(["game_id", "play_id"]).copy()

    # yardline_number is distance to the nearest goal line, measured on `yardline_side`'s half.
    # Raw absolute x of the LOS: own half -> 10 + n (toward the offense's own goal line at x=10),
    # opponent half -> 110 - n. Midfield (n=50) has yardline_side null and lands on x=60 either way.
    own_side = ctx["yardline_side"] == ctx["possession_team"]
    los_if_right = (10 + ctx["yardline_number"]).where(own_side, 110 - ctx["yardline_number"])

    # `los_if_right` assumes the offense drives toward +x; mirror it for plays going left.
    going_left = ctx["play_direction"] == "left"
    ctx["los_x"] = los_if_right.where(~going_left, FIELD_LENGTH - los_if_right)  # raw frame
    ctx["flipped"] = going_left
    # standardized LOS is the offense-driving-right value
    ctx["los_x_std"] = los_if_right
    ctx["yards_to_goal"] = FIELD_LENGTH - 10 - los_if_right
    return ctx[["game_id", "play_id", "flipped", "los_x", "los_x_std", "yards_to_goal"]]


def standardize_tracking(tracking: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    df = tracking.merge(ctx, on=["game_id", "play_id"], how="left", validate="m:1")
    missing = df["flipped"].isna().sum()
    if missing:
        print(f"  warning: {missing:,} tracking rows have no play context (dropped)")
        df = df[df["flipped"].notna()].copy()

    f = df["flipped"].astype(bool)
    df.loc[f, "x"] = FIELD_LENGTH - df.loc[f, "x"]
    df.loc[f, "y"] = FIELD_WIDTH - df.loc[f, "y"]
    # o and dir are compass-style degrees; a 180 degree rotation of the field adds 180.
    for ang in ("o", "dir"):
        df.loc[f, ang] = (df.loc[f, ang] + 180.0) % 360.0

    # position relative to the line of scrimmage: positive = downfield (toward defense's end zone)
    df["x_rel_los"] = df["x"] - df["los_x_std"]
    return df.drop(columns=["flipped", "los_x"])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pp = duckdb.sql(
        f"""select game_id, play_id, possession_team, yardline_side, yardline_number, play_direction
            from read_csv_auto('{RAW / "player_play.csv"}')"""
    ).df()
    ctx = play_context(pp)

    for season in SEASONS:
        print(f"{season}...")
        tracking = pd.read_csv(RAW / f"game_tracking_{season}.csv")
        std = standardize_tracking(tracking, ctx)
        std.to_parquet(OUT / f"game_tracking_std_{season}.parquet", index=False)
        print(f"  wrote {len(std):,} rows")


if __name__ == "__main__":
    main()
