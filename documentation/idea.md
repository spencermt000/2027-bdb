# Spec: Jev-Style Decision Model for NFL Big Data Bowl Tracking Data

Status: DRAFT v0 for manual validation. Nothing here is implemented. Items marked **[VERIFY]** are claims from memory or secondary sources and need checking before you rely on them.

---

## 1. Goal

Build a small, fast, multi-question decision model over NFL tracking data. Input is a spatiotemporal snapshot of a play (one frame, or a short window of frames). Output is a set of typed, calibrated decisions (categories, binary probabilities, scalar distributions), all computed in one forward pass.

The design copies the *pattern* of Jev (TypeSafe AI's "System One" decision model), not the model itself, because Jev takes text input and its weights appear not to be public.

### Non-goals
- Not generating text.
- Not fine-tuning or wrapping Jev itself (see section 2).
- Not a full-play sequence forecaster (possible later extension).

---

## 2. Background and what is known about Jev  [VALIDATED 2026-10-12]

Checked against TypeSafe's docs and blog (docs.typesafe.ai, typesafe.ai/blog) plus secondary reviews. Not read: collapsed FAQ items and evals.typesafe.ai.

Claims and verdicts:
1. **Confirmed.** Input is text only (strings, JSON objects, arrays of text). Output is typed decisions with probabilities, not generated text. Image, audio and video are unsupported.
2. **Mostly confirmed.** Docs: Jev "ingests the `state` once and evaluates every question against it in parallel"; blog: "Generates all outputs in a single query." The literal phrase "single forward pass" is not used. Context limit 64k tokens.
3. **Confirmed.** Caller defines answers via three primitives: Choice (up to 255 options), Score (ordered levels), Noul (true/false). No free-form answers.
4. **Confirmed.** Trained with RLCD (Reinforcement Learning for Calibrated Decisions), explicitly contrasted with RLHF. Algorithm details not published. Calibration is a group-level property, not a per-answer guarantee.
5. **Confirmed (vendor figures).** 70-500 ms end-to-end; $0.042 per million input tokens (= $42 per billion), output tokens free. Latency measured by TypeSafe from a West Coast laptop. TypeSafe says pricing might be subsidized.
6. **NOT confirmed; original numbers likely wrong.** TypeSafe publishes no accuracy percentages, only speed/cost multiples; its own evals use the average of two frontier LLMs as reference. Secondary figures vary by task: vendor-reported ~67.8% vs 73-74% for frontier models; independent tests show 78.8% vs 84.0% (77-class routing), 62.6% vs 81.3% (phishing vs Haiku 4.5), and "level with mid-price LLMs, 6.5-11.5 points behind frontier." Restated claim: roughly mid-tier LLM accuracy, behind frontier, task-dependent.
7. **Confirmed: none offered.** "Jev is not fine-tuned or LoRA-adapted with customer data." No embedding/vector input; endpoints are only `POST /v1/systemone` and `GET /v1/models`. Weights not explicitly stated closed, but API-only in practice. Option A is not viable.

Calibration caveat: independent results conflict. One spam test found ECE 0.053 for Jev vs 0.006 for a frontier model; another found it well calibrated on familiar English tasks and off elsewhere. Selective prediction looked strong (top 90% most confident spam predictions all correct). TypeSafe publishes no ECE, Brier or reliability plots.

Conclusion: a vector input cannot be passed to Jev. The pattern (parallel typed heads, calibration) is reusable. Option B (text-serialized frames) remains a rough baseline only.

Sources: https://docs.typesafe.ai/concepts/system-one, https://docs.typesafe.ai/models, https://docs.typesafe.ai/introduction/machine-learning-primer, https://typesafe.ai/blog/introducing-system-one-models-and-jev

---

## 3. Options considered

| Option | Description | Pros | Cons |
|---|---|---|---|
| A. Adapt Jev | Project frame vectors into Jev's token space, fine-tune | Keeps language prior | Needs weight access and fine-tuning; likely unavailable; language prior has little value for geometry |
| B. Text-serialize frames | Describe the frame in text, call Jev API | Zero training, quick baseline | Numbers tokenize badly; no geometric inductive bias; per-call cost and latency |
| **C. Own encoder + typed heads (recommended)** | Jev's design pattern on a tracking-native encoder | Fast, small, calibrated, trainable on one GPU | Only answers questions it was trained on |
| D. Hybrid | Own encoder produces discrete tags; Jev or an LLM reasons over tags plus context | Allows new questions in plain English | Two systems; tag vocabulary is a bottleneck |

Recommended plan: build C, use B as a baseline to compare against, keep D as a later extension.

---

## 4. Architecture (Option C)

### 4.1 Input representation

**Dataset reality (checked 2026-10-12 against `raw/`):** the game tracking files (`game_tracking_2023/24/25.csv`) cover only the 510-player rookie cohort, not all 22 players. Across all three seasons (8.18M frames, 114,565 plays) the mean is 2.77 tracked players per frame (median 2, max 11), consistent with `player_play.csv`. There is no ball entity and no team/position column in the tracking files. So the original "22 tokens + ball" design is not buildable from this dataset. Revisit if a different year with full tracking is chosen (see section 7).

Per frame, one token per tracked player (variable count, roughly 1-11; pad and mask).

Per-token features available in the tracking files: `x, y, s, a, dis, o, dir` (angles encoded as sin/cos), `event` tag (null on most frames). Joined from `player_play.csv` on `game_id, play_id, nfl_id`:
- Team side (offense / defense, derived from `team_abbr` vs `possession_team`)
- Aligned position (`lined_up_position`, categorical embedding)
- Optionally: height, weight from `combine_results.csv` (attributes only; exclude IDs and names to avoid identity leakage, see section 8)

**TODO (later): direction standardization.** Normalize every play to offense-moves-left-to-right using `play_direction` (in `player_play.csv`, not in the tracking files). For `play_direction == "left"`: rotate 180°, i.e. `x' = 120 - x`, `y' = 53.33 - y`, `dir' = (dir + 180) mod 360`, `o' = (o + 180) mod 360`; `s, a, dis` unchanged. Flipping x alone would mirror the play and is wrong. Fix the direction at the snap for the whole play so turnovers do not flip it mid-play. Also consider x relative to the line of scrimmage (from `yardline_number`, `yardline_side`). [VERIFY the angle convention, assumed 0° = +y, clockwise, 90° = +x, against the data before implementing.]

Play-level context vector (from `player_play.csv`, constant across a play's frames; concatenated or injected as a special token):
- Down, yards to go, yardline, score differential (home/visitor pre-snap scores), game clock, quarter, offense formation, receiver alignment, pre-snap win probability and expected points
- Do not feed `team_coverage_*`, `pass_result`, `yards_gained`, `expected_points_added` or other outcome columns as inputs; they are labels (see 4.4 and section 8)
- Time since snap (frame index relative to snap and to key events like pass_forward, pass_arrived)

Temporal window: configurable K frames (K = 1 baseline, then 5, 10). Options for encoding time: stack features across frames per player, or add a temporal encoder per player track before the spatial encoder. Decision deferred to experiments.

### 4.2 Encoder

Two candidates; compare:
1. **Set transformer / transformer over player tokens** with relative-position bias (pairwise dx, dy, distance, relative velocity as attention bias).
2. **Graph network** over players with edges by proximity or by role (offense-defense, offense-offense, defense-defense). This matches the existing graph-based JEPA work on this data.

Equivariance note: the field has a preferred direction and sidelines, so full rotation invariance is not wanted. Reflection across the field's long axis may be a valid augmentation [VERIFY that it preserves play semantics, e.g., left/right bias].

Output: per-player embeddings H (22 x d) and a pooled frame embedding g (d).

### 4.3 Question tokens and typed heads

- One learned query token per question q_i.
- Cross-attention from each q_i to H (and g), a few layers, all questions in parallel.
- Head type per question:
  - Categorical: softmax over a fixed label set
  - Binary: sigmoid
  - Scalar: Gaussian (mean, log-variance) or quantile outputs, e.g., 10/50/90
  - Per-player questions: head applied to a player's embedding conditioned on q_i (e.g., "will this player be the target?"), softmax across the 22 players

### 4.4 Candidate question set (v1)

Pick a small initial set; labels must be derivable from the data.
1. Route type of target receiver (categorical; from route annotations if available)
2. Defensive coverage shell, man vs zone (categorical or binary; from annotations)
3. Pass completion (binary; play outcome)
4. Target receiver (per-player softmax; needs targeted-receiver label)
5. Yards gained or expected points added (scalar; outcome-derived)
6. Next-k-frames displacement of a given player (scalar vector; labels from later frames)

Final question set to be decided after reading the dataset documentation [VERIFY which labels exist for the chosen season].

---

## 5. Training

### 5.1 Stages
1. **Self-supervised pretraining (optional but recommended):** reuse the existing cross-encoding JEPA approach on this data, or masked-player / masked-future-frame prediction in embedding space. Goal: a general frame encoder.
2. **Supervised multi-task training:** freeze or partially unfreeze the encoder, train query tokens and heads on the labeled questions.
3. **Calibration:** post-hoc temperature scaling per categorical/binary head on a held-out set; check quantile coverage for scalar heads.

### 5.2 Losses
- Categorical / binary: cross-entropy / BCE (proper scoring rules)
- Scalar: Gaussian NLL or pinball loss
- Multi-task weighting: start with uncertainty-based weighting or equal weights; tune

### 5.3 Splits
- Split by **game** (and ideally by week or team), never by frame or play, to avoid leakage from near-duplicate frames within a play.
- Hold out a separate season if multiple are used.

---

## 6. Evaluation

Accuracy metrics per question:
- Categorical: accuracy, macro-F1, log loss
- Binary: AUROC, log loss, Brier score
- Scalar: MAE/RMSE, CRPS or pinball loss, interval coverage

Calibration metrics:
- Expected calibration error, reliability diagrams
- **Selective prediction:** accuracy vs coverage curve when abstaining below a confidence threshold. This tests the "act when confident, escalate otherwise" premise.

Baselines:
1. Majority class / simple context-only model (gradient boosted trees on tabular context features)
2. Flat MLP over concatenated player features
3. Option B: text-serialized frames to Jev or a frontier LLM (cost and latency tracked)
4. Per-question dedicated models vs the shared multi-question model (does sharing help or hurt?)

Latency and size targets (initial, adjustable): under 10 ms per frame on GPU, under 5M parameters for the first version [these are guesses; set after profiling].

Ablations: temporal window K, encoder type (transformer vs graph), with and without pretraining, with and without context features.

---

## 7. Data

- Source: NFL Big Data Bowl 2027 (Kaggle), already in `raw/`: 510-player rookie cohort (2023-25 drafts), Combine tracking (`combine_tracking.csv`, `combine_results.csv`), in-game tracking for 2023-25 (`game_tracking_*.csv`), `player_play.csv` (~314k rows, ~115k plays), `games.csv`, `players.csv`, `player_career_successes.csv`. The year is therefore decided. Game tracking covers only the rookie cohort (see 4.1). Other BDB years have different schemas and were not checked [VERIFY if switching].
- The competition itself asks for a framework linking 10 Hz Combine tracking to future NFL performance, judged on write-ups. This play-level model is not that task; decide whether it is a personal project or a competition entry.
- License: 2027 rules limit competition data to non-commercial use (competition, forums, academic research and education). Data license appears to be CC BY-NC 4.0 (seen on the 2026 data page and a third-party 2027 repo) [VERIFY on the 2027 Kaggle Data tab]. Winners must open-source their submission under an OSI-approved license (separate from the data terms). Rules bar private sharing of code or data outside a team. Do not commit raw data to a public repo; check the rules before sharing derived artifacts such as checkpoints. No post-competition retention clause found [VERIFY].
- Preprocessing: standardize field direction, convert angles, handle missing frames, align frames to events (snap, pass_forward, pass_arrived, tackle).
- Sample unit: (play, frame) with its question labels. Class imbalance expected for some questions.

---

## 8. Risks and open questions

1. **Partial observability (new, biggest risk).** Tracking covers only the rookie cohort: ~2.8 tracked players per frame on average (median 2, max 11), no ball, no opponents unless they are also rookies. Play-level labels (coverage, completion, EPA) must be inferred from a few players, so achievable accuracy may be low and mostly driven by the context vector. The context-only tabular baseline (section 6) is therefore the key comparison: if tracking adds nothing over context, the encoder is not earning its place.
2. **Selection bias (new).** Plays are included only when a rookie was on the field, so the data over-represents certain teams, positions, and situations. Results may not generalize to all plays.
3. **Label availability.** Resolved for the 2027 data (see 4.4): coverage man/zone and shell (~88% non-null), pass result (null on run plays), yards gained and EPA are usable; target receiver (rookie is target on only 13.9% of pass plays) and route type (`route_ran` null on 83% of rows) are not. Open: class imbalance, e.g. PREVENT has ~633 rows.
4. **Leakage.** Later-frame information (e.g., pass_arrived, tackle events, final positions) can leak outcomes into early frames. Define the input window cutoff per question; the `event` column itself is a leak source. Also keep outcome columns (`team_coverage_*`, `pass_result`, `yards_gained`, EPA) out of the context vector.
5. **Identity leakage.** Using `nfl_id` or names lets the model memorize players or plays; exclude them. With only ~510 players, memorization is easier than with a full league, so consider holding out players as well as games.
6. **Frame-level vs play-level decisions.** Per-frame is simpler to train; per-play needs history and a clear decision time. The main labels are play-level, which favors per-play with a decision cutoff (e.g., at snap or at pass_forward). Unresolved.
7. **Is the Jev-pattern worth it?** The shared multi-question single-pass design may not beat dedicated models. The ablation in section 6 answers this.
8. **Whether the language-prior hybrid (Option D) is needed at all.** Defer until C is working.
9. **Dependence on prior work.** How much of the existing JEPA NFL code (graph construction, encoder) can be reused is unchecked. It lives at `/root/jepa-nfl-bdb/` on yomp, which was unreachable (SSH timeout) when checked on 2026-10-12, and no local copy was found. Also check whether it assumed full 22-player tracking; if so, the graph construction will not transfer to rookie-only data.
10. **Project purpose.** Undecided whether this is a personal project or a competition entry; the 2027 competition task is Combine-to-career performance, not play-level decisions (see section 7).

---

## 9. Milestones

1. Validate section 2 claims and choose dataset year and question set (manual).
2. Data pipeline: normalized frames, event alignment, labels, game-level splits.
3. Baselines: tabular model and flat MLP; establish numbers.
4. Encoder + typed heads, single frame, 2-3 questions.
5. Add temporal window, more questions, calibration, selective-prediction analysis.
6. Optional: JEPA pretraining; Option B baseline against Jev API; hybrid extension.

---

## 10. Checklist for manual validation

- [x] Jev: input type, output type, parallel questions, weights/fine-tuning availability, pricing, latency, benchmark claims
- [x] Big Data Bowl year chosen (2027); columns and labels listed (see 4.1, 4.4, 7)
- [ ] Field-direction normalization (to-do, see 4.1) and reflection augmentation are valid
- [ ] License and terms for the data (partly checked, confirm on Kaggle Data tab)
- [x] Which questions have trustworthy labels (see 4.4 and section 8)
- [ ] Reusable pieces from the existing JEPA NFL code (yomp unreachable on 2026-10-12)
- [ ] Parameter and latency targets reasonable