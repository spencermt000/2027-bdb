"""Generate the EDA notebooks with nbformat. Run from repo root:

    python eda/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace eda/0*.ipynb
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent

SETUP = """import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from utils import *
set_style()
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 160)"""


def nb(title, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    n.cells = [nbf.v4.new_markdown_cell(title)]
    n.cells.append(nbf.v4.new_code_cell(SETUP))
    for kind, src in cells:
        n.cells.append((nbf.v4.new_markdown_cell if kind == "md" else nbf.v4.new_code_cell)(src.strip()))
    return n


def md(s): return ("md", s)
def code(s): return ("code", s)


# --------------------------------------------------------------------------- 01
n1 = nb("# 01 · Players & career outcomes\n\nFiles: `players.csv`, `player_career_successes.csv`.\n\n"
        "**Questions:** Who is in the 510-player cohort (position, draft class, college)? How are draft slot and career outcomes related? How sparse are the accolades?", [
    code("""players = load_csv("players", parse_dates=["birth_date"])
career = load_csv("player_career_successes")
print(players.shape, career.shape)
assert players.nfl_id.is_unique and career.nfl_id.is_unique
print("ids match:", set(players.nfl_id) == set(career.nfl_id))
players.head()"""),
    md("## Missingness"),
    code("missingness(players)"),
    md("Nulls in the draft columns should correspond exactly to undrafted free agents (UDFAs)."),
    code("""players["udfa"] = players.draft_overall_pick.isna()
print(players.groupby("draft_year").udfa.agg(["sum", "mean", "size"]))
assert (players.draft_round.isna() == players.udfa).all()"""),
    md("## Cohort composition"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
order = players.nfl_position.value_counts().index
sns.countplot(data=players, y="nfl_position", order=order, color=PALETTE[0], ax=axes[0]); axes[0].set_title("Players by NFL position")
sns.countplot(data=players, x="draft_year", hue="udfa", ax=axes[1], palette=[PALETTE[0], PALETTE[1]]); axes[1].set_title("Draft class: drafted vs UDFA")
plt.tight_layout(); savefig(fig, "01_composition")"""),
    code("""fig, ax = plt.subplots(figsize=(8, 6))
conf = players.college_conference.value_counts()
sns.barplot(x=conf.values, y=conf.index, color=PALETTE[0], ax=ax); ax.set_title("Players by college conference")
savefig(fig, "01_conference")"""),
    code("""players["age_at_draft"] = (pd.to_datetime(players.draft_year.astype(str) + "-04-25") - players.birth_date).dt.days / 365.25
fig, ax = plt.subplots()
sns.histplot(players.age_at_draft, bins=25, color=PALETTE[0], ax=ax); ax.set_title("Age at draft (years)")
players.age_at_draft.describe()"""),
    md("## Draft capital"),
    code("""drafted = players[~players.udfa]
fig, ax = plt.subplots()
sns.histplot(drafted.draft_overall_pick, bins=26, color=PALETTE[0], ax=ax); ax.set_title("Overall pick (drafted players only)")
print(drafted.draft_round.value_counts().sort_index())"""),
    md("## Career outcomes"),
    code("""df = players.merge(career, on="nfl_id")
df["total_snaps"] = df.career_offensive_snaps + df.career_defensive_snaps + df.career_special_teams_snaps
describe_num(career.drop(columns="nfl_id"))"""),
    code("""acc = ["ap_all_pro_1st_team", "ap_all_pro_2nd_team", "pro_bowl_original_ballot"]
print("share of players with >=1:"); print((career[acc] > 0).mean().round(3))
print("no snaps at all:", (df.total_snaps == 0).sum(), "| zero games active:", (df.career_games_active == 0).sum())"""),
    code("""df["pick_filled"] = df.draft_overall_pick.fillna(260)   # UDFAs plotted past the last pick
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(data=df, x="pick_filled", y="career_games_started", hue="draft_year", palette=PALETTE[:3], s=18, ax=axes[0]); axes[0].set_title("Games started vs overall pick (260 = UDFA)")
sns.scatterplot(data=df, x="pick_filled", y="total_snaps", hue="draft_year", palette=PALETTE[:3], s=18, legend=False, ax=axes[1]); axes[1].set_title("Career snaps vs overall pick")
plt.tight_layout(); savefig(fig, "01_pick_vs_outcomes")"""),
    code("""df["round"] = df.draft_round.fillna(8).astype(int).replace({8: "UDFA"}).astype(str)
fig, ax = plt.subplots()
sns.boxplot(data=df, x="round", y="career_games_started", order=[str(i) for i in range(1, 8)] + ["UDFA"], color=PALETTE[0], fliersize=2, linewidth=1, ax=ax)
ax.set_title("Games started by draft round")
df.groupby("draft_year")[["career_games_active", "career_games_started", "total_snaps"]].median()"""),
    md("**Note:** later draft classes have had less time to accumulate games, so career totals are **not comparable across `draft_year`** without exposure adjustment (e.g. per-game rates)."),
    md("## Findings / follow-ups\n- Fill in after reviewing outputs."),
])

# --------------------------------------------------------------------------- 02
n2 = nb("# 02 · Combine results\n\nFile: `combine_results.csv` (joined to `players.csv`).\n\n"
        "**Questions:** Which measurables are missing and why? How do they differ by position? How do they relate to each other and to draft slot?", [
    code("""comb = load_csv("combine_results")
players = load_csv("players")
print(comb.shape, "unique ids:", comb.nfl_id.nunique())
comb.head()"""),
    code("missingness(comb)"),
    md("## Missingness by position group"),
    code("""drills = ["hand_size","arm_length","wing_span","ten_yd_split","forty","vertical","broad_jump","three_cone","short_shuttle","bench_reps"]
miss = comb.groupby("combine_position")[drills].agg(lambda s: s.isna().mean()).round(2)
miss["n"] = comb.combine_position.value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(miss[drills], annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "share missing"}, ax=ax); ax.set_title("Share missing by combine position")
savefig(fig, "02_missing")
miss"""),
    md("## Distributions by position"),
    code("""meas = ["combine_height","combine_weight","forty","vertical","broad_jump","three_cone","short_shuttle","bench_reps"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.flat, meas):
    sns.histplot(comb[c].dropna(), bins=25, color=PALETTE[0], ax=ax); ax.set_title(c)
plt.tight_layout(); savefig(fig, "02_dists")"""),
    code("""order = comb.groupby("combine_position").forty.median().sort_values().index
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.boxplot(data=comb, x="combine_position", y="forty", order=order, color=PALETTE[0], fliersize=2, linewidth=1, ax=axes[0]); axes[0].set_title("40-yard dash by combine position")
sns.boxplot(data=comb, x="combine_position", y="combine_weight", order=order, color=PALETTE[0], fliersize=2, linewidth=1, ax=axes[1]); axes[1].set_title("Weight by combine position")
plt.tight_layout(); savefig(fig, "02_by_position")"""),
    md("## Relationships"),
    code("""num = comb[["combine_height","combine_weight","hand_size","arm_length","wing_span","ten_yd_split","forty","vertical","broad_jump","three_cone","short_shuttle","bench_reps","ngs_athleticism_score","ngs_college_production_score","ngs_final_score"]]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(num.corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax); ax.set_title("Pairwise correlation (pairwise-complete)")
savefig(fig, "02_corr")"""),
    code("""fig, ax = plt.subplots()
sns.scatterplot(data=comb, x="ten_yd_split", y="forty", s=14, color=PALETTE[0], ax=ax); ax.set_title("10-yard split vs 40-yard dash")
print("rows where 10yd split >= forty:", (comb.ten_yd_split >= comb.forty).sum())"""),
    md("## NGS scores vs draft capital"),
    code("""df = comb.merge(players[["nfl_id","draft_overall_pick","draft_round","nfl_position"]], on="nfl_id")
df["udfa"] = df.draft_overall_pick.isna()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, ["ngs_athleticism_score","ngs_college_production_score","ngs_final_score"]):
    sns.scatterplot(data=df, x="draft_overall_pick", y=c, hue="udfa", palette=[PALETTE[0], PALETTE[1]], s=14, legend=(c == "ngs_final_score"), ax=ax); ax.set_title(c)
plt.tight_layout(); savefig(fig, "02_ngs_vs_pick")
df[["draft_overall_pick","ngs_athleticism_score","ngs_college_production_score","ngs_final_score"]].corr(method="spearman").round(2)"""),
    md("## Sanity checks"),
    code("""print("combine_position vs nfl_position mismatch (top):")
pd.crosstab(df.combine_position, df.nfl_position).stack().loc[lambda s: s > 0].sort_values(ascending=False).head(15)"""),
    code("""z = (num - num.mean()) / num.std()
out = z.abs().stack().loc[lambda s: s > 4]
print(len(out), "values with |z| > 4"); out.sort_values(ascending=False).head(15)"""),
    md("## Findings / follow-ups\n- Fill in after reviewing outputs."),
])

# --------------------------------------------------------------------------- 03
n3 = nb("# 03 · Combine tracking\n\nFile: `combine_tracking.csv` (10 Hz optical tracking, ~6.3k drill attempts).\n\n"
        "**Questions:** What drills/attempts exist? Is the tracking clean (frame cadence, gaps, nulls)? Does tracking-derived 40 time agree with the official one?", [
    code("""ct = load_csv("combine_tracking", parse_dates=["time"])
print(ct.shape); ct.head()"""),
    code("missingness(ct)"),
    code("""print(ct.entity_type.value_counts())
ev = ct.groupby("event_id").agg(draft_year=("draft_year","first"), drill_type=("drill_type","first"), drill_name=("drill_name","first"),
                                nfl_id=("nfl_id","first"), attempt=("attempt","first"), n=("time","size"), t0=("time","min"), t1=("time","max"))
ev["dur"] = (ev.t1 - ev.t0).dt.total_seconds()
print("events:", len(ev))
ev.head()"""),
    md("## Coverage"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
ev.drill_type.value_counts().plot.barh(color=PALETTE[0], ax=axes[0]); axes[0].set_title("Events by drill_type"); axes[0].invert_yaxis()
ev.groupby(["draft_year","drill_type"]).size().unstack(0).plot.barh(ax=axes[1], color=PALETTE[:3]); axes[1].set_title("Events by drill_type and year"); axes[1].invert_yaxis()
plt.tight_layout(); savefig(fig, "03_coverage")"""),
    code("""print(ev.drill_name.value_counts().to_string())
print(); print("players:", ev.nfl_id.nunique(), "| attempts per player per drill:"); ev.groupby(["nfl_id","drill_name"]).size().describe()"""),
    md("## Tracking quality"),
    code("""ct = ct.sort_values(["event_id","time"])
ct["gap"] = ct.groupby(["event_id","entity_type"]).time.diff().dt.total_seconds()
print(ct.gap.round(2).value_counts().head(8))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(ev.dur, bins=40, color=PALETTE[0], ax=axes[0]); axes[0].set_title("Event duration (s)")
sns.boxplot(data=ev, y="drill_type", x="dur", color=PALETTE[0], fliersize=2, linewidth=1, ax=axes[1]); axes[1].set_title("Duration by drill_type")
plt.tight_layout()"""),
    code("""p = ct[ct.entity_type == "PLAYER"]
print(p[["x","y","s","a","dis"]].describe(percentiles=[.001,.5,.999]).T)
print("out of field bounds:", ((p.x < 0) | (p.x > 120) | (p.y < 0) | (p.y > 53.34)).mean())"""),
    md("## 40-yard dash"),
    code("""forty = p[p.drill_type == "FORTY_YARD_DASH"].copy()
print("40-yd events:", forty.event_id.nunique())
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for eid in forty.event_id.drop_duplicates().sample(25, random_state=0):
    e = forty[forty.event_id == eid]; t = (e.time - e.time.min()).dt.total_seconds()
    axes[0].plot(t, e.s, color=PALETTE[0], alpha=.4, lw=1); axes[1].plot(t, e.x - e.x.iloc[0], color=PALETTE[0], alpha=.4, lw=1)
axes[0].set_title("Speed (yd/s) vs time, 25 random dashes"); axes[1].set_title("Distance along field vs time")
plt.tight_layout(); savefig(fig, "03_forty_curves")"""),
    code("""def split_time(e, dist=40/3):
    # time until first frame where displacement along x reaches `dist` yards, from first moving frame
    d = (e.x - e.x.iloc[0]).abs().values; t = (e.time - e.time.iloc[0]).dt.total_seconds().values
    start = t[np.argmax(d > 0.5)] if (d > 0.5).any() else np.nan
    hit = np.argmax(d >= dist) if (d >= dist).any() else None
    return np.nan if hit is None else t[hit] - start
res = forty.groupby(["event_id","nfl_id","draft_year"]).apply(split_time).rename("track_forty").reset_index()
best = res.groupby("nfl_id").track_forty.min().reset_index()
comb = load_csv("combine_results")[["nfl_id","forty"]].merge(best, on="nfl_id")
fig, ax = plt.subplots(figsize=(5.5, 5))
sns.scatterplot(data=comb, x="forty", y="track_forty", s=14, color=PALETTE[0], ax=ax); ax.plot([4, 6], [4, 6], color=MUTED, lw=1); ax.set_title("Official vs tracking-derived 40 (rough)")
print(comb[["forty","track_forty"]].corr().iloc[0,1], "n =", len(comb))"""),
    md("The tracking-derived time here is a crude threshold-crossing estimate (it is not the official timer start/finish), so expect a constant offset; correlation is the useful signal."),
    md("## Agility drill paths"),
    code("""drills = p[p.drill_type.isin(["THREE_CONE_DRILL","SHORT_SHUTTLE"])]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, dt in zip(axes, ["THREE_CONE_DRILL","SHORT_SHUTTLE"]):
    d = drills[drills.drill_type == dt]
    for eid in d.event_id.drop_duplicates().sample(min(15, d.event_id.nunique()), random_state=0):
        e = d[d.event_id == eid]; ax.plot(e.x - e.x.iloc[0], e.y - e.y.iloc[0], lw=1, alpha=.6, color=PALETTE[0])
    ax.set_title(dt); ax.set_aspect("equal")
plt.tight_layout(); savefig(fig, "03_agility_paths")"""),
    md("## Findings / follow-ups\n- Fill in after reviewing outputs."),
])

# --------------------------------------------------------------------------- 04
n4 = nb("# 04 · Games & player-play\n\nFiles: `games.csv`, `player_play.csv` (316k player-play rows for the rookie cohort).\n\n"
        "**Questions:** How are snaps distributed across players/positions/games? What context and which role-specific metrics are populated? How do plays join to games and players?", [
    code("""games = load_csv("games")
pp = load_csv("player_play")
players = load_csv("players")
print(games.shape, pp.shape)
pp.head(3)"""),
    md("## Games"),
    code("""print(games.groupby(["season","season_type"]).size().unstack())
fig, ax = plt.subplots()
games[games.season_type == "REG"].groupby(["season","week"]).size().unstack(0).plot(ax=ax, color=PALETTE[:3]); ax.set_title("Regular-season games per week"); ax.set_ylim(0)
assert games.game_id.is_unique"""),
    md("## Join integrity"),
    code("""print("player_play games in games.csv:", pp.game_id.isin(games.game_id).mean())
print("player_play nfl_ids in players.csv:", pp.nfl_id.isin(players.nfl_id).mean(), "| players with >=1 play:", pp.nfl_id.nunique(), "/", len(players))
print("duplicate (game, play, player):", pp.duplicated(["game_id","play_id","nfl_id"]).sum())
pp = pp.merge(games[["game_id","season","season_type","week"]], on="game_id", how="left")"""),
    code("missingness(pp).head(45)"),
    md("Nulls mostly reflect role-specific metrics (they only exist for linemen / rushers / targets). Below: non-null share of metric families by `lined_up_position`."),
    code("""fam = {"receiving": ["target","rec_yards","route_ran","separation_at_pass_forward","cushion"],
       "pass_pro": ["pass_rushers_encountered","pressure_allowed","sack_allowed","time_to_pressure_allowed"],
       "pass_rush": ["player_get_off","time_to_pressure","quick_pressure","unblocked_pressure"],
       "coverage": ["coverage_assignment","coverage_assignment_at_snap"], "rushing": ["rush_yards"]}
cov = pd.DataFrame({k: pp.groupby("lined_up_position")[v].apply(lambda g: g.notna().mean().mean()) for k, v in fam.items()})
cov["n"] = pp.lined_up_position.value_counts()
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cov.drop(columns="n").sort_values("pass_pro"), annot=True, fmt=".2f", cmap="Blues", ax=ax); ax.set_title("Mean non-null share by metric family and lined_up_position")
savefig(fig, "04_role_coverage")"""),
    md("## Snaps"),
    code("""snaps = pp.groupby("nfl_id").size()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(snaps, bins=40, color=PALETTE[0], ax=axes[0]); axes[0].set_title("Player-play rows per player")
pp.lined_up_position.value_counts().plot.barh(color=PALETTE[0], ax=axes[1]).invert_yaxis(); axes[1].set_title("Rows by lined_up_position")
plt.tight_layout(); savefig(fig, "04_snaps")
print(pp.groupby("season").agg(rows=("play_id","size"), players=("nfl_id","nunique"), games=("game_id","nunique")))"""),
    md("## Situation context"),
    code("""fig, axes = plt.subplots(2, 3, figsize=(15, 7))
pp.down.value_counts().sort_index().plot.bar(color=PALETTE[0], ax=axes[0,0]).set_title("down")
pp.quarter.value_counts().sort_index().plot.bar(color=PALETTE[0], ax=axes[0,1]).set_title("quarter")
sns.histplot(pp.yards_to_go.clip(upper=25), bins=25, color=PALETTE[0], ax=axes[0,2]).set_title("yards_to_go (clipped 25)")
pp.offense_formation.value_counts().plot.barh(color=PALETTE[0], ax=axes[1,0]).set_title("offense_formation")
pp.team_coverage_type.value_counts().plot.barh(color=PALETTE[0], ax=axes[1,1]).set_title("team_coverage_type")
pp.pass_result.value_counts(dropna=False).plot.bar(color=PALETTE[0], ax=axes[1,2]).set_title("pass_result (NaN = non-pass)")
plt.tight_layout(); savefig(fig, "04_context")"""),
    md("## EPA / win probability"),
    code("""u = pp.drop_duplicates(["game_id","play_id"])
print("unique plays:", len(u)); print(u[["expected_points_added","expected_points","home_team_win_probability_added","yards_gained"]].describe(percentiles=[.01,.5,.99]).T)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(u.expected_points_added.clip(-6, 6), bins=60, color=PALETTE[0], ax=axes[0]); axes[0].set_title("EPA per play (clipped ±6)")
sns.histplot(u.yards_gained.clip(-10, 40), bins=50, color=PALETTE[0], ax=axes[1]); axes[1].set_title("yards_gained (clipped)")
plt.tight_layout()"""),
    code("""print(pp.play_nullified_by_penalty.value_counts(normalize=True).round(4))
print("home+visitor WPA sums to ~0:", np.allclose(pp.home_team_win_probability_added + pp.visitor_team_win_probility_added, 0, atol=1e-6))"""),
    md("## Role-specific distributions"),
    code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(pp.separation_at_pass_forward.dropna(), bins=40, color=PALETTE[0], ax=axes[0]).set_title("separation_at_pass_forward")
sns.histplot(pp.player_get_off.dropna(), bins=40, color=PALETTE[0], ax=axes[1]).set_title("player_get_off")
sns.histplot(pp.time_to_pressure_allowed.dropna(), bins=40, color=PALETTE[0], ax=axes[2]).set_title("time_to_pressure_allowed")
plt.tight_layout(); savefig(fig, "04_role_metrics")"""),
    code("""pp["is_target"] = pp.target.fillna(False).astype(bool)
rec = pp[pp.route_ran.notna()]
print("target rate by route:"); print(rec.groupby("route_ran").is_target.agg(["mean","size"]).sort_values("size", ascending=False).round(3))"""),
    md("## Findings / follow-ups\n- Fill in after reviewing outputs."),
])

# --------------------------------------------------------------------------- 05
n5 = nb("# 05 · Game tracking (sample)\n\nFiles: `game_tracking_{2023,2024,2025}.csv` — 0.3–1.0 GB each. Uses a **5% game-level sample** per season cached to `eda/cache/` (see `utils.sample_tracking`).\n\n"
        "**Questions:** Schema/cadence/quality of the tracking; how it joins to `player_play`; what the full-data size/strategy implies.", [
    code("""import time
frames = {}
for s in (2023, 2024, 2025):
    t = time.time(); frames[s] = sample_tracking(s, frac_games=0.05); print(s, frames[s].shape, f"{time.time()-t:.1f}s")
gt = pd.concat([f.assign(season_file=s) for s, f in frames.items()], ignore_index=True)
gt["time"] = pd.to_datetime(gt.time)
gt.head()"""),
    code("""print("games sampled:", gt.groupby("season_file").game_id.nunique().to_dict())
print("memory MB:", round(gt.memory_usage(deep=True).sum()/1e6, 1))
missingness(gt)"""),
    md("## Structure: rows per play, players per play, frame cadence"),
    code("""gt = gt.sort_values(["game_id","play_id","nfl_id","time"])
gt["gap"] = gt.groupby(["game_id","play_id","nfl_id"]).time.diff().dt.total_seconds()
print(gt.gap.round(2).value_counts().head(6))
per = gt.groupby(["game_id","play_id","nfl_id"]).size()
print("frames per (play, player):"); print(per.describe())
print("tracked players per play:"); print(gt.groupby(["game_id","play_id"]).nfl_id.nunique().describe())"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(per.clip(upper=200), bins=50, color=PALETTE[0], ax=axes[0]); axes[0].set_title("Frames per play-player (clipped)")
gt.event.value_counts().plot.barh(color=PALETTE[0], ax=axes[1]).invert_yaxis(); axes[1].set_title("Event tags")
plt.tight_layout(); savefig(fig, "05_structure")"""),
    code("""gt["month"] = gt.time.dt.to_period("M")
print(pd.crosstab(gt.month, gt.season_file))"""),
    md("Season files can contain Jan–Feb timestamps of the following calendar year (late-season/playoff games); the file label is the NFL season."),
    md("## Kinematics & quality"),
    code("""print(gt[["x","y","s","a","dis","o","dir"]].describe(percentiles=[.001,.5,.999]).T)
print("out of bounds share:", ((gt.x < 0) | (gt.x > 120) | (gt.y < 0) | (gt.y > 53.34)).mean())
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(gt.s.sample(200000, random_state=0), bins=60, color=PALETTE[0], ax=axes[0]).set_title("speed (yd/s)")
sns.histplot(gt.a.clip(-10, 10).sample(200000, random_state=0), bins=60, color=PALETTE[0], ax=axes[1]).set_title("acceleration (clipped)")
sns.histplot(gt.dis.sample(200000, random_state=0), bins=60, color=PALETTE[0], ax=axes[2]).set_title("dis per frame")
plt.tight_layout(); savefig(fig, "05_kinematics")"""),
    md("## Speed by position (join to players & player_play)"),
    code("""players = load_csv("players")[["nfl_id","nfl_position"]]
pp = load_csv("player_play", usecols=["game_id","play_id","nfl_id","lined_up_position"])
keys = gt[["game_id","play_id","nfl_id"]].drop_duplicates()
m = keys.merge(pp, on=["game_id","play_id","nfl_id"], how="left", indicator=True)
print("tracking (play,player) found in player_play:", (m._merge == "both").mean())
print("tracking nfl_ids in players.csv:", keys.nfl_id.isin(players.nfl_id).mean())"""),
    code("""top = gt.merge(players, on="nfl_id").groupby(["game_id","play_id","nfl_id","nfl_position"]).s.max().rename("max_speed").reset_index()
order = top.groupby("nfl_position").max_speed.median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.boxplot(data=top, x="nfl_position", y="max_speed", order=order, color=PALETTE[0], fliersize=1.5, linewidth=1, ax=ax); ax.set_title("Max in-play speed (yd/s) by NFL position")
savefig(fig, "05_speed_by_pos")"""),
    md("## Sample of one play"),
    code("""g, p = gt[gt.event == "ball_snap"][["game_id","play_id"]].iloc[0]
play = gt[(gt.game_id == g) & (gt.play_id == p)]
fig, ax = plt.subplots(figsize=(9, 4))
for i, (nid, e) in enumerate(play.groupby("nfl_id")): ax.plot(e.x, e.y, color=PALETTE[i % 8], lw=2, marker="o", ms=2)
ax.set_xlim(0, 120); ax.set_ylim(0, 53.3); ax.set_aspect("equal"); ax.set_title(f"Tracked rookies on game {g} play {p}")"""),
    md("## Scale & full-data strategy\nSizes on disk: 2023 ≈ 0.34 GB, 2024 ≈ 0.72 GB, 2025 ≈ 1.02 GB (≈2.1 GB). Use the row counts below (extrapolated from the 5% sample) to decide among DuckDB-on-CSV, Parquet conversion, or polars."),
    code("""import os
rows = []
for s in (2023, 2024, 2025):
    size = os.path.getsize(RAW / f"game_tracking_{s}.csv") / 1e9
    rows.append({"season": s, "csv_GB": round(size, 2), "sample_rows": len(frames[s]), "est_total_rows_M": round(len(frames[s]) / 0.05 / 1e6, 1),
                 "est_in_memory_GB": round(frames[s].memory_usage(deep=True).sum() / 0.05 / 1e9, 1)})
pd.DataFrame(rows)"""),
    md("## Findings / follow-ups\n- Fill in after reviewing outputs."),
])

for name, n in {"01_players_and_outcomes": n1, "02_combine_results": n2, "03_combine_tracking": n3,
                "04_games_and_player_play": n4, "05_game_tracking_sample": n5}.items():
    nbf.write(n, OUT / f"{name}.ipynb")
    print("wrote", name)
