"""Shared helpers for the EDA notebooks."""
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
CACHE = Path(__file__).resolve().parent / "cache"
FIGURES = Path(__file__).resolve().parent / "figures"

# Categorical palette, fixed order (light-mode steps from the dataviz reference palette).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

NA_VALUES = ["NA", ""]


def set_style():
    """Thin marks, recessive grid, no top/right spines."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left",
        "lines.linewidth": 2, "figure.dpi": 110, "figure.figsize": (8, 4),
    })


def savefig(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")


def load_csv(name, **kw):
    """Read raw/<name>.csv, treating 'NA' as missing."""
    return pd.read_csv(RAW / f"{name}.csv", na_values=NA_VALUES, **kw)


def missingness(df):
    """Per-column null count / share, sorted descending."""
    n = df.isna().sum()
    out = pd.DataFrame({"n_missing": n, "pct_missing": (n / len(df) * 100).round(2), "dtype": df.dtypes.astype(str)})
    return out.sort_values("n_missing", ascending=False)


def describe_num(df):
    return df.describe(percentiles=[.01, .25, .5, .75, .99]).T


def sample_tracking(season, frac_games=0.05, seed=0):
    """Game-level sample of game_tracking_<season>.csv, cached as parquet.

    Samples whole games (hash of game_id) so plays stay intact. Uses duckdb so the
    CSV is streamed, never fully loaded. Swap this body when the full-data strategy
    (duckdb / polars / parquet) is decided.
    """
    CACHE.mkdir(exist_ok=True)
    tag = f"{frac_games:g}_{seed}"
    path = CACHE / f"game_tracking_{season}_sample_{tag}.parquet"
    if not path.exists():
        mod = max(int(round(1 / frac_games)), 1)
        src = RAW / f"game_tracking_{season}.csv"
        duckdb.sql(
            f"""COPY (
                  SELECT * FROM read_csv('{src}', nullstr='NA')
                  WHERE hash(game_id + {seed}) % {mod} = 0
                ) TO '{path}' (FORMAT parquet)"""
        )
    return pd.read_parquet(path)
