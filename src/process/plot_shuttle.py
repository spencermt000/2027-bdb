"""Plot real shuttle attempts with detected plants and phases.

Usage: python src/process/plot_shuttle.py [event_id ...]   (default: a seeded sample of QC-passing attempts)
Output: outputs/figures/shuttle_phases.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import paths, pipeline_cfg  # noqa: E402

PHASE_COLORS = ["#9ecae1", "#fb6a4a", "#a1d99b", "#fb6a4a", "#fdd0a2"]
PHASE_NAMES = ["leg1", "plant1", "leg2", "plant2", "leg3"]


def plot_attempts(att: pd.DataFrame, fr: pd.DataFrame, event_ids: list[str], path):
    n = len(event_ids)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n), squeeze=False)
    for r, eid in enumerate(event_ids):
        a = att[att["event_id"] == eid].iloc[0]
        f = fr[fr["event_id"] == eid]
        axp, axv = axes[r]
        for pid in range(5):
            m = f["phase_id"] == pid
            if m.any():
                for ax in (axp, axv):
                    ax.axvspan(f.loc[m, "t"].min(), f.loc[m, "t"].max(), color=PHASE_COLORS[pid], alpha=0.35, lw=0)
        axp.plot(f["t"], f["p_rel"], "k")
        axp.fill_between(f["t"], f["p_rel"] - 2 * np.sqrt(f["p_var"]), f["p_rel"] + 2 * np.sqrt(f["p_var"]), color="k", alpha=0.15)
        axv.plot(f["t"], f["v_rel"], "k", label="v")
        axv.plot(f["t"], f["a_rel"] / 4, color="tab:purple", lw=0.8, label="a / 4")
        axv.axhline(0, color="gray", lw=0.5)
        for k in (1, 2):
            for ax in (axp, axv):
                ax.axvline(a[f"t_plant_{k}"], color="r", ls="--", lw=0.8)
                ax.axvline(a[f"t_brake_onset_{k}"], color="b", ls=":", lw=0.8)
        for ax in (axp, axv):
            ax.axvline(a["t_start"], color="g", lw=0.8)
            if np.isfinite(a["t_end"]):
                ax.axvline(a["t_end"], color="g", lw=0.8)
        axp.set_ylabel("p_rel (yd)")
        axv.set_ylabel("v_rel (yd/s)")
        axp.set_title(f"{eid[-18:]}  t_total={a['t_total']:.2f}  official={a['short_shuttle']}  {a['qc_reason'] or 'ok'}", fontsize=8)
        if r == n - 1:
            axp.set_xlabel("t (s)")
            axv.set_xlabel("t (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in PHASE_COLORS]
    fig.legend(handles, PHASE_NAMES, loc="upper center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = paths()
    att = pd.read_parquet(p["derived"] / "shuttle_attempt_features.parquet")
    fr = pd.read_parquet(p["derived"] / "shuttle_frames.parquet")
    ids = sys.argv[1:] or list(att[att["qc_pass"]].sample(5, random_state=pipeline_cfg()["seed"])["event_id"])
    out = p["outputs"] / "figures" / "shuttle_phases.png"
    plot_attempts(att, fr, ids, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
