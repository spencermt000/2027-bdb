"""Run the evaluation harness over whatever experiments have written attempt_scores.parquet.

Writes outputs/comparison_report.md. Usage: python src/process/run_harness.py
"""
import pandas as pd

from config import paths, pipeline_cfg
from harness import build_report, load_all_scores
from loaders import load_results


def main() -> None:
    cfg = pipeline_cfg()
    p = paths()
    cond = pd.read_parquet(p["derived"] / "conditioning.parquet")
    scores = load_all_scores(p["outputs"])
    text = build_report(scores, cond, load_results(), cfg["eval"], cfg["seed"])
    (p["outputs"] / "comparison_report.md").write_text(text)
    print(f"wrote {p['outputs'] / 'comparison_report.md'} ({len(scores)} experiments)")


if __name__ == "__main__":
    main()
