"""Stage B driver: shuttle attempt features and frame table.

Needs build_forty.py to have run first (reads data/derived/forty_reference.parquet).
Outputs: data/derived/shuttle_attempt_features.parquet, data/derived/shuttle_frames.parquet
Usage:   python src/process/build_shuttle.py
"""
import pandas as pd

from config import paths, pipeline_cfg, seed_everything
from loaders import load_results, load_tracking
from shuttle import build_shuttle


def qc_report(att: pd.DataFrame, fr: pd.DataFrame, cfg: dict) -> str:
    q = [.05, .25, .5, .75, .95]
    both = att[att["t_total"].notna() & att["short_shuttle"].notna()]
    two = att["n_plants_found"] == 2
    mid = both.assign(t_mid=both["t_plant_2"] - both["t_plant_1"])
    by_year = both.groupby("draft_year").apply(lambda d: d["t_total"].corr(d["short_shuttle"], method="spearman"), include_groups=False)
    cols = ["t_total", "t_leg1", "t_leg2", "t_leg3", "leg_dist_1", "leg_dist_2", "leg_dist_3", "entry_speed_1", "entry_speed_2",
            "brake_onset_dist_1", "brake_onset_dist_2", "brake_time_1", "brake_time_2", "peak_decel_1", "peak_decel_2",
            "dwell_1", "dwell_2", "reaccel_ratio_1", "reaccel_ratio_2", "entry_ratio_1", "entry_ratio_2", "cod_cost", "finish_speed"]
    L = ["# Stage B: shuttle QC", "", f"Config: `{cfg}`", "",
         f"- Attempts: {len(att)}, players: {att['nfl_id'].nunique()}; frame-table rows: {len(fr)}",
         f"- Exactly two plants found: {int(two.sum())} / {len(att)} ({two.mean():.1%}); one plant: {int((att['n_plants_found'] == 1).sum())}; "
         f"too few frames: {int(att['n_plants_found'].isna().sum())}",
         f"- Plant features usable (`qc_plants_ok`): {int(att['qc_plants_ok'].sum())}",
         f"- Finish observed (`t_total` defined): {int(att['t_total'].notna().sum())} (the clip ends before the finish line otherwise)",
         f"- Full QC pass: {int(att['qc_pass'].sum())} ({att['qc_pass'].mean():.1%})", "",
         "## QC flags", "", "```", att["qc_reason"].replace("", "ok").str.split(",").explode().value_counts().to_string(), "```", "",
         "## t_total vs official `short_shuttle`", "",
         f"n = {len(both)}. Median offset (tracked - official): {both['resid_t_total'].median():.2f} s; "
         f"SD of centered residual (IQR-based): {(both['resid_t_total_c'].quantile(.75) - both['resid_t_total_c'].quantile(.25)) / 1.349:.3f} s.", "",
         f"- Spearman, all: {both['t_total'].corr(both['short_shuttle'], method='spearman'):.3f}; Pearson: {both['t_total'].corr(both['short_shuttle']):.3f}",
         "- Spearman by draft year: " + ", ".join(f"{k}: {v:.3f}" for k, v in by_year.items()),
         f"- Spearman of the middle-leg time (`t_plant_2 - t_plant_1`) with official: {mid['t_mid'].corr(mid['short_shuttle'], method='spearman'):.3f}",
         f"- Spearman of `t_leg1` with official: {both['t_leg1'].corr(both['short_shuttle'], method='spearman'):.3f}; "
         f"`t_leg3`: {both['t_leg3'].corr(both['short_shuttle'], method='spearman'):.3f}", "",
         "```", att[["resid_t_total", "resid_t_total_c"]].describe(percentiles=q).round(3).T.to_string(), "```", "",
         "## Feature summary (QC-passing attempts)", "", "```", att.loc[att["qc_pass"], cols].describe(percentiles=q).round(2).T.to_string(), "```", "",
         "## Start direction", "", "```", att["start_dir"].value_counts(dropna=False).to_string(), "```", "",
         "Players with 2+ passing attempts and mixed start direction: "
         f"{int((att[att['qc_pass']].groupby('nfl_id')['start_dir'].nunique() > 1).sum())}", ""]
    return "\n".join(L)


def main() -> None:
    cfg = pipeline_cfg()
    seed_everything(cfg["seed"])
    p = paths()
    ref = pd.read_parquet(p["derived"] / "forty_reference.parquet")
    att, fr = build_shuttle(load_tracking(["SHORT_SHUTTLE"]), load_results(), ref, cfg["shuttle"])
    att.to_parquet(p["derived"] / "shuttle_attempt_features.parquet", index=False)
    fr.to_parquet(p["derived"] / "shuttle_frames.parquet", index=False)
    (p["outputs"] / "shuttle_qc.md").write_text(qc_report(att, fr, cfg["shuttle"]))
    print(f"{len(att)} attempts, {int(att['qc_pass'].sum())} pass QC; {len(fr)} frames -> {p['derived']}")


if __name__ == "__main__":
    main()
