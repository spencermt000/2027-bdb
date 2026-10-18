# CLAUDE.md — BDB 2027 Short-Shuttle Evaluation (Specs 1.0–1.5)

Instructions for Claude Code to build the code for the shared pipeline (spec 1.0) and five scoring experiments (T1–T5, specs 1.1–1.5). The specs live in this Obsidian vault under `2027-bdb-research/03-specs/` and are the source of truth; this file summarizes them and sets the working rules. If this file and a spec disagree, the spec wins. Flag the conflict instead of picking one silently.

| Spec | File | Builds |
|---|---|---|
| 1.0 | `2027-bdb-research/03-specs/1.0 Shared Pipeline & Evaluation Harness.md` | Forty fit, shuttle segmentation, feature tables, bio table, evaluation harness |
| 1.1 | `2027-bdb-research/03-specs/1.1 T1 Hierarchical Bayesian Model on Attempt Features.md` | T1: hierarchical Bayesian model on attempt features |
| 1.2 | `2027-bdb-research/03-specs/1.2 T2 Probabilistic Next-k-Frame Forecasting.md` | T2: GRU + mixture density network next-k-frame forecaster |
| 1.3 | `2027-bdb-research/03-specs/1.3 T3 Switching State-Space Model (AR-HMM SLDS).md` | T3: input-driven sticky AR-HMM |
| 1.4 | `2027-bdb-research/03-specs/1.4 T4 Symbolic Regression for Braking and CoD Laws.md` | T4: PySR symbolic regression |
| 1.5 | `2027-bdb-research/03-specs/1.5 T5 Self-Supervised Trajectory Encoder + Conditional Flow.md` | T5: contrastive trajectory encoder + conditional normalizing flow |

Read the relevant spec in full before starting each milestone.

---

## 1. Goal in one paragraph

Score every NFL Combine **short shuttle (5-10-5) attempt** relative to what is expected for that player's body and sprint profile. The sprint profile comes from fitting a model to the player's **40-yard dash tracking**. Five methods produce attempt-level scores in a shared output format, and one harness compares them on reliability, incremental validity, held-out fit and cross-method agreement. The unit of analysis is a single attempt `(nfl_id, event_id, attempt)`. Never average attempts before modelling; attempt-to-attempt differences are a signal of interest.

---

## 2. Working rules

- **Stop at each milestone checkpoint** (section 8). Report what was built, test results and anything surprising in the data, then wait for a go-ahead before starting the next milestone.
- **Ask before** adding a dependency not listed in section 5, changing an output schema, or resolving an item in "Open decisions" (section 9).
- **Data is read-only.** Never modify files in the raw data directory. Never commit raw data or derived parquet files to git. The competition data is licensed for non-commercial use only.
- **No leakage.** The exclusions in section 4.3 are hard rules. Add a test that fails if an excluded column reaches any model input.
- **Grouped splits everywhere.** Every train/validation split is grouped by `nfl_id`. No window, frame or attempt from one player may appear on both sides.
- **Thresholds are config, not constants.** Every threshold in the specs (start detection, braking onset, dwell speed, QC limits) goes in a YAML config and gets logged with each run.
- **Seeds fixed and logged.** One global seed in config; record it in every output.
- **Units.** Tracking is in yards and yd/s. Convert to SI (m, m/s, kg) only where the spec requires (`F0`, `P_max`, `brake_impulse`). Keep the conversion in one module (`src/common/units.py`): 1 yd = 0.9144 m, 1 lb = 0.45359237 kg.
- **Unverified claims.** If something in a spec looks wrong or the data contradicts it, note it in the milestone report rather than working around it quietly.

---

## 3. Data

The raw data location is set in `configs/paths.yaml` (`raw_dir`), overridable by the `BDB_RAW_DIR` environment variable. Do not hard-code paths.

### 3.1 Files used

| File | Use |
|---|---|
| `combine_tracking.csv` | Frames with `drill_type in {FORTY_YARD_DASH, SHORT_SHUTTLE}` and `entity_type == "PLAYER"` |
| `combine_results.csv` | Bio and measurements. Official `forty`, `ten_yd_split`, `short_shuttle` for **validation only** |
| `players.csv` | `nfl_position`, `draft_year`. Draft columns are loaded but never used as model inputs |

### 3.2 Relevant columns

- `combine_tracking.csv`: `draft_year, event_id, nfl_id, entity_type, time (ISO 8601, 10 Hz), drill_type, drill_name, attempt, x, y, s, a, dis, dir`. There is **no `o` (orientation)**. Rows with `entity_type == "BALL"` carry the player's `nfl_id`; drop them.
- `combine_results.csv`: `nfl_id, draft_year, combine_position, combine_height (in), combine_weight (lb), hand_size, arm_length, wing_span, ten_yd_split, forty, three_cone, short_shuttle, bench_reps, vertical, broad_jump, ngs_athleticism_score, ngs_college_production_score, ngs_final_score`. Drill columns contain nulls for opt-outs.
- `players.csv`: `nfl_id, display_name, draft_year, nfl_position, birth_date, college_name, college_conference, draft_round, draft_pick_within_round, draft_overall_pick`.

Field coordinates: `x` 0–120 yd, `y` 0–53.33 yd. Drill direction varies, so every attempt is projected onto its own principal axis (spec 1.0, Stage A step 2 and Stage B step 1).

### 3.3 First task with the data

Before writing pipeline code, write `scripts/inspect_data.py` that reports and saves to `outputs/data_inspection.md`:

- Attempts per player per drill (forty and shuttle), and how many players have 1, 2 or 3+ shuttle attempts.
- Frames per attempt, gaps in `time` (dropped frames), duplicate timestamps.
- Whether `attempt` and `event_id` are one-to-one.
- Null rates for the bio columns and official drill times.
- `drill_name` values under `SHORT_SHUTTLE` and `FORTY_YARD_DASH`.
- Start direction (L/R) counts per player, for the open decision in section 9.

This is the first checkpoint.

---

## 4. Shared pipeline (spec 1.0)

### 4.1 Stage A: 40-yard dash, per attempt (`src/forty/`)

1. Group frames by `event_id`, sort by `time`.
2. Project `(x, y)` onto the first principal component to get 1D position `p(t)`, oriented so the run is positive.
3. Detect start: first frame of a sustained rise in `s` (default `s > 0.3 yd/s` for 3+ frames). Back-extrapolate `t0` from the fit rather than trusting the threshold frame.
4. Detect end: first frame where `p - p_start >= 40 yd`, sub-frame time by linear interpolation.
5. Fit `v(t) = v_max * (1 - exp(-(t - t0)/tau))` by nonlinear least squares on smoothed speed. Free parameters: `v_max, tau, t0`.
6. Derive per attempt: `v_max, tau, a0 = v_max/tau, F0 = mass_kg * a0_mps2, P_max = F0 * v_max_mps / 4, t90, d90, speed_drop, fit_rmse, t10_model, t40_model`.
7. QC: compare `t10_model`/`t40_model` with official `ten_yd_split`/`forty`. Flag high `fit_rmse` or implausible residuals. Report the discrepancy distribution.
8. Player-level forty reference: mean of each parameter over QC-passing attempts (default). Also store best-attempt values for a sensitivity check.

Provide `x_ref(t) = v_max * (t - tau * (1 - exp(-t/tau)))` (t from `t0`) and its numerical inverse `t_ref(d)`, used by `cod_cost` and `entry_ratio`.

Output: `data/derived/forty_attempts.parquet`, `data/derived/forty_reference.parquet`.

### 4.2 Stage B: shuttle, per attempt (`src/shuttle/`)

1. Group by `event_id`, project onto the principal axis to get signed `p(t)` and `v(t)`.
2. Smooth with an RTS smoother using a constant-acceleration (or jerk-limited) model on `(x, y)`. Output smoothed position, velocity, acceleration, **each with variance**. Smoothing spline is an acceptable fallback.
3. Segment: plants are the two minima of `|v|` where `v` changes sign. Phases: `leg1, plant1, leg2, plant2, leg3`. QC-flag attempts without exactly two plants. Record start direction.
4. Attempt feature table `data/derived/shuttle_attempt_features.parquet` with the columns in spec 1.0 Stage B step 4 (`t_total, t_leg{1,2,3}, entry_speed_{1,2}, brake_onset_dist_{1,2}, brake_dist_{1,2}, brake_time_{1,2}, mean_decel_{1,2}, peak_decel_{1,2}, brake_impulse_{1,2}, dwell_{1,2}, reaccel_{1,2}, reaccel_ratio_{1,2}, entry_ratio_{1,2}, cod_cost, cod_cost_plant_{1,2}, start_dir, attempt_order`, plus a QC flag).
   - `cod_cost = t_total - [t_ref(5) + t_ref(10) + t_ref(5)]`.
   - Validate `t_total` against official `short_shuttle`.
5. Frame table `data/derived/shuttle_frames.parquet`: smoothed `p, v, a` (+ variances), signed distance to the next turn line `d_line`, `phase_id`, `t_since_start`, frame index. Used by T2, T3, T5.

### 4.3 Stage C: bio table and conditioning vector (`src/data/bio.py`)

- **Inputs:** `combine_height, combine_weight, arm_length, hand_size, wing_span, combine_position, nfl_position, ngs_college_production_score`.
- **Excluded from all model inputs (hard rule):**
  - `ngs_athleticism_score`, `ngs_final_score` (likely include shuttle results; leakage check only)
  - official `short_shuttle` (target-adjacent)
  - `three_cone` (out of scope)
  - all draft variables (`draft_round, draft_pick_within_round, draft_overall_pick`)
- Missing values: keep, add missingness indicators. Opt-outs are not random.
- Conditioning vector `c` per attempt = player-level forty reference + bio table, standardized (scaler fit on training folds only). Position one-hot, or a random effect in T1.

### 4.4 Evaluation harness (`src/eval/`)

1. Splits: `GroupKFold(n_splits=5)` by `nfl_id`, fixed seed; save fold assignments to `data/derived/folds.parquet` so every experiment uses the same folds. Secondary: leave-one-`draft_year`-out.
2. Baseline: LightGBM quantile models (q = 0.1, 0.5, 0.9) predicting `t_total` from `c`. Score = attempt's quantile position. Written in the standard output format as `exp_id = "baseline"`.
3. Reliability: ICC of attempt-level scores for players with 2+ attempts; Spearman of player rankings attempt 1 vs attempt 2.
4. Incremental validity: partial correlation / R² gain of each score over `t_total` and the forty reference.
5. Held-out fit: predictive log-likelihood on held-out players where defined (T1, T2, T5).
6. Cross-experiment agreement: Spearman matrix of player-level scores across T1–T5 and baseline; list the players with largest disagreement.
7. Leakage check: correlation of each score with `ngs_athleticism_score` (report it; high is expected).

The harness reads every `outputs/<exp_id>/attempt_scores.parquet` present and writes `outputs/comparison_report.md`. It must run with any subset of experiments finished.

### 4.5 Output convention (all experiments)

`outputs/<exp_id>/attempt_scores.parquet`, one row per attempt:

| Column | Type | Notes |
|---|---|---|
| `nfl_id` | int | |
| `event_id` | str | |
| `attempt` | int | |
| `exp_id` | str | `baseline`, `T1` … `T5` |
| `score_primary` | float | signed, **higher = better** |
| `score_*` | float | experiment-specific extras |
| `score_uncertainty` | float, nullable | if available |

Write a schema validator in `src/eval/schema.py` and call it at the end of every experiment.

---

## 5. Stack

- Python 3.11, managed with `uv` (`pyproject.toml`, lockfile committed).
- Data: `polars` (or `pandas` where a library needs it), `pyarrow`, `numpy`, `scipy`.
- Classical ML: `scikit-learn`, `lightgbm`.
- T1: `numpyro` + `jax` (PyMC acceptable fallback), `arviz` for diagnostics and PSIS-LOO.
- T2, T5: `torch`. T5 flow: `zuko` (`nflows` fallback). Write the TS2Vec-style encoder in-repo rather than depending on a package.
- T3: `dynamax` (JAX); `ssm` fallback.
- T4: `pysr`. It needs a Julia install; check for it and fail with a clear message if missing. Isolate it so the rest of the repo runs without Julia.
- Plotting: `matplotlib`. Optional: `umap-learn` for T5 visuals.
- Dev: `pytest`, `ruff`, `pre-commit`. Configs: plain YAML + a small typed loader (dataclasses or pydantic); no Hydra unless asked.

Make heavy dependencies optional extras (`uv sync --extra t1`, `--extra t3`, etc.) so the shared pipeline installs light.

---

## 6. Repo layout

```
CLAUDE.md
pyproject.toml
configs/
  paths.yaml          # raw_dir, derived_dir, outputs_dir
  pipeline.yaml       # thresholds, smoother params, QC limits, seed
  t1.yaml … t5.yaml
src/
  common/             # units, config loading, logging, seeding
  data/               # loaders, filtering, axis projection, smoothing (RTS), bio table, conditioning vector
  forty/              # start/end detection, sprint fit, QC, reference
  shuttle/            # segmentation, attempt features, frame table
  eval/               # folds, baseline, reliability, validity, agreement, leakage, schema, report
experiments/
  T1/ T2/ T3/ T4/ T5/ # model, train/fit script, scoring; each has run.py
scripts/
  inspect_data.py
  build_pipeline.py   # Stages A–C end to end
  run_harness.py
tests/
data/derived/         # gitignored
outputs/              # gitignored except *.md reports if wanted
```

Each experiment has a single entry point: `uv run python -m experiments.T1.run --config configs/t1.yaml`.

---

## 7. Experiment summaries

Full details are in each spec. These are the parts most easily missed.

### T1 — Hierarchical Bayesian (spec 1.1)
- Targets modelled separately first: `cod_cost` (attempt level); `brake_dist, dwell, reaccel_ratio, entry_ratio` (plant level, long format with `plant_idx`, `entry_speed` as covariate except for `entry_ratio`).
- Student-t likelihood; player and position random effects; `attempt_order` slope γ.
- Variants: V1 linear; V2 per-player random slope on `entry_speed`; V3 joint multivariate with LKJ prior.
- NUTS, 4 chains × 1000 warmup × 1000 draws; require R̂ < 1.01, report ESS.
- Outputs: `score_primary` = sign-flipped standardized residual of `cod_cost`; `score_pit_<target>`; `player_effect_<target>` (mean, SD); variance decomposition (model ICC), γ, coefficients, PIT calibration.
- Held-out players: draw `u_i` from its prior.

### T2 — Next-k-frame forecasting (spec 1.2)
- Sliding windows, n past → k future frames; translation-invariant targets (`Δp`, `v`).
- GRU encoder (1–2 layers, hidden 64), FiLM conditioning on `c`, MDN head (K = 3).
- Sweeps: n ∈ {2, 3, 5, 10, 20}, k ∈ {1, 3, 5, 10}, conditioning ∈ {none, forty, bio, forty+bio}.
- Baselines: constant velocity, constant acceleration, linear AR, model without `c`.
- Scores: per-frame signed z, phase means, braking-onset timing deviation, attempt NLL. `score_primary` = mean signed z over plant + re-acceleration phases (more distance than predicted = better).
- Report NLL vs n (the "previous frames" question), NLL vs k, 50%/90% interval coverage.
- Window weighting so long attempts don't dominate. Never split windows of one attempt across train/val.

### T3 — Switching state-space (spec 1.3)
- Sticky AR-HMM (order 1–2) on smoothed `v`, `a`; main model is input-driven (transition logits depend on `d_line` and `c`; state biases depend on `c`). SLDS only if needed.
- K ∈ 3..8, EM with multiple restarts; pick smallest K within 1 SE of best held-out log-likelihood.
- Validate states against Stage B phases: boundary error (ms) and adjusted Rand index.
- Outputs: Viterbi paths to `outputs/T3/state_sequences.parquet`; `dur_<state>`, `n_switch`, `brake_onset_dline`, within-state metrics, `ll_attempt`; expectation-relative `score_*` from a conditional model on `c`.
- `score_primary` = sign-flipped residual of time in brake + plant + re-accel states.
- Report state stability across restarts.

### T4 — Symbolic regression (spec 1.4)
- Targets (separate runs): `brake_dist`, `brake_time`, `plant_cost` (plant level), `cod_cost` (attempt level).
- PySR with `+ − × / pow`, `square sqrt exp log`, maxsize ≈ 20, Huber loss, units declared; dimensionally consistent forms reported separately.
- Run within each grouped fold; select at the Pareto knee on held-out loss; keep only forms that recur across folds and 20 player-level bootstraps; refit constants with bootstrap CIs.
- Reference models: physics prior `entry_speed² / (2b)` (cohort `b`, and `b ∝ F0/mass`), linear regression, LightGBM.
- Outputs: residual scores, `outputs/T4/equations.json`, Pareto plots, stability table.
- Watch collinearity: `F0 = mass · a0` makes equivalent forms look different.

### T5 — SSL encoder + conditional flow (spec 1.5)
- Resample each attempt to L = 64, two variants: time-normalized and distance-normalized. Channels: signed `v`, `a`, `|v|`, `d_line`, plus absolute duration as a separate scalar.
- TS2Vec-style dilated 1D conv encoder, embedding 16–32, InfoNCE. Positive pairs: augmentations (small time-warp, noise matched to smoother variance, ±5% amplitude, ≥90% crop); variant: same player's other attempt; hierarchical/temporal contrast.
- Checks: linear probes (`t_total`, `cod_cost`, plant features, `mass` as a leakage probe), nearest neighbours, seed stability.
- Stage 2: PCA to 6–8 dims (fit on training folds), conditional neural spline flow / MAF p(e | c); baseline conditional Gaussian; also unconditional flow.
- Scores: `ll_attempt`, `pct_attempt`, directional `score_primary` (projection of `e − E[e|c]` onto the probe direction for `−cod_cost`), `score_style_<k>`.
- Typicality is not quality; report it as "how unusual" only.
- Optional cross-drill pretraining is a scope decision: ask before doing it.

---

## 8. Build order and checkpoints

Stop and report after each numbered milestone.

1. **Scaffold + data inspection.** Repo layout, `pyproject.toml`, config loader, loaders, `scripts/inspect_data.py` and its report.
2. **Stage A (forty).** Fit, QC report (model vs official times), player reference. Tests on synthetic sprints with known `v_max`, `tau`.
3. **Stage B (shuttle).** Smoother, segmentation, attempt features, frame table, QC report (`t_total` vs official `short_shuttle`; share of attempts with exactly two plants). Tests on synthetic 5-10-5 traces with known plants. Include a few plots of real attempts with detected phases.
4. **Stage C + harness + baseline.** Bio table, conditioning vector, folds, LightGBM baseline, schema validator, report generator run on the baseline alone. Leakage test.
5. **T1.** V1 first, then V2, then V3.
6. **T4.** (Lightweight, uses only the attempt table; good second experiment.)
7. **T3.**
8. **T2.**
9. **T5.**
10. **Full comparison report** across all experiments.

T1–T5 are independent once milestone 4 is done; this order goes from cheapest to most expensive. Change it only if asked.

---

## 9. Open decisions (do not resolve without asking)

- Forty reference: player-level mean (default) vs best attempt. Implement both; run the sensitivity check.
- Thresholds for start detection, braking onset and dwell speed: expose as config and report sensitivity.
- Whether to model start direction (L vs R) as a covariate: decide after the milestone-1 counts.
- T5 cross-drill pretraining on all Combine drills.
- T5 positive-pair variant 1 vs 2 as the default.

---

## 10. Testing and quality

- Unit tests for: axis projection (sign and orientation), sprint fit on synthetic data, `t_ref` inversion, plant detection on synthetic shuttles, RTS smoother on a known trajectory, fold grouping (no `nfl_id` overlap), leakage guard, output schema.
- Integration test: run Stages A–C and the baseline on a small fixture (a handful of players, checked into `tests/fixtures/` as **synthetic** data, never real rows).
- `ruff check` and `pytest` must pass before each checkpoint report.
- Every experiment report (`outputs/<exp_id>/report.md`) states: config used, seed, git commit, number of attempts/players after QC, and runtime.

---

## 11. Vault notes

The Obsidian vault holds research notes, not code. Do not edit vault notes unless asked. When a milestone produces a result worth recording (e.g. QC findings, chosen K for T3, discovered equations), say so in the checkpoint report so it can be logged in `04-experiments/` or `06-decisions/`.
