"""Raw-data loaders for the combine short-shuttle / forty pipeline. Raw files are read-only."""
import pandas as pd

from config import paths

DRILLS = ("FORTY_YARD_DASH", "SHORT_SHUTTLE")


def load_tracking(drills=DRILLS) -> pd.DataFrame:
    """Player frames for the given drills, sorted by (event_id, time). BALL rows are dropped."""
    df = pd.read_csv(paths()["raw"] / "combine_tracking.csv")
    df = df[df["drill_type"].isin(drills) & (df["entity_type"] == "PLAYER")].copy()
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values(["event_id", "time"]).reset_index(drop=True)


def load_results() -> pd.DataFrame:
    return pd.read_csv(paths()["raw"] / "combine_results.csv", na_values=["NA"])


def load_players() -> pd.DataFrame:
    return pd.read_csv(paths()["raw"] / "players.csv", na_values=["NA"])
