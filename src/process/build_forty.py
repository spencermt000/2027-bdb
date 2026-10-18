"""Stage A driver: fit every forty attempt, run QC, write the reference.

Outputs: data/derived/forty_attempts.parquet, data/derived/forty_reference.parquet,
         outputs/forty_qc.md
Usage:   python src/process/build_forty.py
"""
import pandas as pd

from config import paths, pipeline_cfg, seed_everything
from forty import PARAMS, build_forty
from loaders import load_results, load_tracking


def qc_report(att: pd.DataFrame, ref: pd.DataFrame, cfg: dict) -> str:
    q = [.05, .25, .5, .75, .95]
    L = ["# Stage A: forty QC", "", f"Config: `{cfg}`", "",
         f"- Attempts: {len(att)}, players: {att['nfl_id'].nunique()}",
         f"- QC pass: {int(att['qc_pass'].sum())} ({att['qc_pass'].mean():.1%})",
         f"- Attempts already in motion on the first frame (start detection fires at frame 0): {att['starts_in_motion'].mean():.1%}",
         f"- Attempts reaching 40 yd by the fit's start line: {att['reached_40'].mean():.1%}",
         f"- Players with a reference: {len(ref[ref['n_pass'] > 0])} of {len(ref)}",
         "", "## QC flags", "",
         "```", att["qc_reason"].replace("", "ok").str.split(",").explode().value_counts().to_string(), "```", "",
         "## Model vs official times (seconds; model - official)", "",
         "The model clock starts at motion onset (`t0`); the official clock evidently starts later, "
         "so a near-constant positive offset is expected. QC thresholds the offset-centered residual.", "",
         "```", att[["resid_t10", "resid_t40", "resid_t10_c", "resid_t40_c"]].describe(percentiles=q).round(3).T.to_string(), "```", "",
         f"Official times missing (cannot be QC'd on time): forty {int(att['forty'].isna().sum())}, ten_yd_split {int(att['ten_yd_split'].isna().sum())}.", "",
         f"Correlation model vs official: t40 {att['t40_model'].corr(att['forty']):.3f}, t10 {att['t10_model'].corr(att['ten_yd_split']):.3f}", "",
         "## Fit parameters (QC-passing attempts)", "",
         "```", att.loc[att["qc_pass"], PARAMS + ["fit_rmse"]].describe(percentiles=q).round(3).T.to_string(), "```", ""]
    both = ref.dropna(subset=["n_pass"]).query("n_pass >= 2")
    if len(both):
        ok = att[att["qc_pass"]]
        within = ok.groupby("nfl_id")["t40_model"].std().mean()
        L += ["## Reference sensitivity", "",
              f"Players with 2+ passing attempts: {len(both)}. Mean within-player SD of `t40_model`: {within:.3f} s. "
              f"Mean |mean - best| `t40_model`: {(both['t40_model'] - both['best_t40_model']).abs().mean():.3f} s.", ""]
    return "\n".join(L)


def main() -> None:
    cfg = pipeline_cfg()
    seed_everything(cfg["seed"])
    p = paths()
    p["derived"].mkdir(parents=True, exist_ok=True)
    p["outputs"].mkdir(parents=True, exist_ok=True)
    att, ref = build_forty(load_tracking(["FORTY_YARD_DASH"]), load_results(), cfg["forty"])
    att.to_parquet(p["derived"] / "forty_attempts.parquet", index=False)
    ref.to_parquet(p["derived"] / "forty_reference.parquet", index=False)
    (p["outputs"] / "forty_qc.md").write_text(qc_report(att, ref, cfg["forty"]))
    print(f"{len(att)} attempts, {int(att['qc_pass'].sum())} pass QC, {len(ref)} players -> {p['derived']}")


if __name__ == "__main__":
    main()
