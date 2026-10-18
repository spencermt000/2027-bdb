"""Grouped splits shared by every experiment. Folds are by nfl_id: no player appears on both sides."""
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_folds(players: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """`players`: one row per nfl_id with `n_attempts` and `draft_year`. GroupKFold with a fixed shuffle seed.

    Returns nfl_id, fold (0..n_splits-1), draft_year.
    """
    players = players.sort_values("nfl_id").reset_index(drop=True)
    fold = pd.Series(-1, index=players.index)
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(gkf.split(players, groups=players["nfl_id"])):
        fold.iloc[test] = k
    return pd.DataFrame({"nfl_id": players["nfl_id"], "fold": fold, "draft_year": players["draft_year"]})


def check_grouped(df: pd.DataFrame, fold_col: str = "fold") -> None:
    """Raise if any nfl_id sits in more than one fold."""
    n = df.groupby("nfl_id")[fold_col].nunique()
    if (n > 1).any():
        raise AssertionError(f"players in multiple folds: {list(n[n > 1].index)[:5]}")
