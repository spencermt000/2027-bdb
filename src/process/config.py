"""Config and path loading. Raw data location comes from configs/paths.yaml or BDB_RAW_DIR."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def load_yaml(name: str) -> dict:
    with open(CONFIGS / name) as f:
        return yaml.safe_load(f)


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def paths() -> dict[str, Path]:
    cfg = load_yaml("paths.yaml")
    raw = os.environ.get("BDB_RAW_DIR", cfg["raw_dir"])
    return {
        "raw": _resolve(raw),
        "derived": _resolve(cfg["derived_dir"]),
        "outputs": _resolve(cfg["outputs_dir"]),
    }


def pipeline_cfg() -> dict:
    return load_yaml("pipeline.yaml")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
