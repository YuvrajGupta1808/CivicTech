# RideScore DC — Ordinal Crash-Risk: build plan (Claude coordinates, Sonnet subagents build)

## Context
- The hackathon is today and submissions close at 16:00 EDT. The plan was written at 13:59, which leaves about 2 hours. `~/hackathon` is empty, so we start from zero.
- **User decisions:**
  - Claude coordinates and Sonnet subagents (`model: "sonnet"`) build pieces in parallel.
  - The project is standalone, with `~/hackathon` as the root of a **new public GitHub repo, `YuvrajGupta1808/CivicTech`**.
  - **No PR to ridescoredc-models.**
  - The labels come from DC's crash data; Claude decides how to use OSM and DDOT.
- **Following the AVTCA research:**
  - Use the AVB-Engage ordinal tooling:
    - expected-level threshold decoding fitted on validation
    - inverse-sqrt class weighting
    - fixed-weight late fusion
    - a ward-independent split
    - every result read against the majority-class baseline
  - Lay the repo out like AVTCA-Research: the four `docs/` files, `AGENTS.md`, `src/`, `scripts/`, `tests/`.
  - **M4 (OSM↔DDOT cross-attention) returns as a time-boxed parallel track.** It is fused in only if it is ready by 15:15; otherwise the GBM ladder is the result.
- **Goal:** answer RQ1, RQ2, RQ3 (E6, plus E7/E8 if M4 lands), RQ4 (decoding) and RQ5 (the surprise map). Hand in a per-segment Parquet, a README and the content for 2 slides, all pushed to the public repo.

## Facts verified today (these override the pasted doc)
- **Crashes (MapServer/24):**
  - 3,089 bike crashes in the window `REPORTDATE >= DATE '2021-09-30' AND REPORTDATE < DATE '2026-09-30'`. Pages hold 1,000 rows; paginate with `resultOffset`.
  - Keys are 32-character lowercase hex.
  - 142 crashes have the sentinel `'Route not found'` instead of a key. Count them as unlocated.
  - `OFFINTERSECTION` is in **metres**.
- **Snapshot:** 28,978 segments → **19,554 sub-blocks**. `dc_*` values are constant within a sub-block, so we can safely keep one row per sub-block.
- **Join:** 88.8% of crashes match a `dc_subblockkey` (93.0% excluding the sentinel). Wards 7 and 8 match only 77%, which goes into the limitations. **Go with sub-blocks.**
- **Labels:** 0 = 17,826 (91.2%), 1 = 1,203 (6.2%), 2+ = 525 (2.7%).
  - Level-2 counts by ward: W1 104, W2 185, W3 **12**, W4 37, W5 68, W6 87, W7 **11**, W8 21.
- **Data quirks:**
  - `dc_SPEEDLIMITS_IB` is 100% null, so use `dc_SPEEDLIMITS_OB` and fall back to `_OB_ALT`.
  - These columns are all null; drop them: `dc_LEFTTURN_EXCLUSIVE`, `dc_RIGHTTURN_EXCLUSIVE`, `dc_*CURBLANE_EXCL`, `dc_MIDMEASURE`.
  - Where `dc_AADT` is present (42% of matched rows), its year is 2020 in every case, so `aadt_is_2020` carries no information. Drop it and note the problem in the limitations.
  - `osm_cycleway = "crossing"` (1,513 rows) marks a crossing, not a lane; map it to `none`. `osm_highway = cycleway` has 3,134 rows of separately mapped tracks.
- **Environment:**
  - `python3` (conda, 3.13) already has pandas 3.0, sklearn 1.8, pyarrow, numpy, scipy, torch-cpu and pytest.
  - **Missing:** geopandas, statsmodels, matplotlib, pyyaml. `uv` is not installed.
- **Reusable code:**
  - AVB-Engage `src/engine/calibration.py`: `expected_scores`, `fit_expected_thresholds`, `refine_expected_thresholds`, `fit_logit_bias`, `apply_logit_bias`, `decode_argmax`, `ordinal_metrics`. It works for any number of classes. Swap `_score_candidate` (lines 128–136) from accuracy to **macro-F1**. Port the tests from `tests/test_calibration.py`.
  - AVB-Engage `scripts/fullclip/gbm_member.py:41-45`: the GBM recipe.
  - AVB-Engage `scripts/fullclip/fuse_members.py` and `scripts/paper/springer_analysis.py`: patterns for fusion and weight sensitivity.
  - ridescoredc-models (`develop`): `ridescore.models.ridescore_v1.lts.lts_level(facility, speed, lanes, function)`, plus the `ridescore.network.normalise` helpers (`classify_facility`, `speed_limit`, `num_lanes`, `name_function`, `segment`). With these, **M1 is the production LTS v1**, which also covers E2. Install with pip from git; if that fails, vendor `lts.py` and `normalise.py` with attribution.
  - De facto hand-in format (from PR #14): one row per segment, keyed by `osm_u, osm_v, osm_key` plus `snapshot_date="2026-09-29"`.

## Design decisions (changes to the pasted doc)
- **Split:** leave-one-ward-out.
  - Test = ward w; validation = the **next 2 wards in cyclic order**, so every fold has at least 32 level-2 validation sub-blocks; train = the other 5 wards.
  - Each sub-block gets one **out-of-fold (OOF)** prediction, from the fold where its ward is the test ward. Maps, the surprise analysis and the hand-in all use these.
- **Models:**
  - **M0:** majority class.
  - **M1:** production LTS v1. LTS 1–2 → 0, 3 → 1, 4 → 2; the score is the LTS level.
  - **M2:** SPF, a negative binomial with offset log(length).
  - **M3:** GBM, the AVB-Engage recipe with sample weights 1/√count.
  - **M4 (stretch):** the cross-attention Transformer from doc §12.5, with d=32, 4 heads, 2 exchanges, source dropout 0.15, ordinal loss λ=0.15 with label smoothing 0.1, and 3 seeds. Reuse AVTCA `models/transformer.py` (`AttentionBlock`, `Mlp`) and `AttentionPool` (`models/multimodal_cnn.py:12-29`), and `OrdinalDistanceCrossEntropy` (`src/engine/runtime.py:443-458`).
  - **F:** 0.67·M3 + 0.33·M2. If M4 lands, use 0.4·M4 + 0.4·M3 + 0.2·M2 (the doc's weights).
- **GBM settings:** `early_stopping=False`, as in AVB-Engage. Missing values are handled natively, so there are no blanket `*_missing` columns. Categoricals use the pandas `category` dtype with `categorical_features="from_dtype"`.
- **Length:** use `dc_LENGTH` (metres, native to the sub-block); fall back to the summed road-segment length in EPSG:26985.
- **Intersection cutoff** for label variant B: 15 m (`OFFINTERSECTION <= 15` counts as an intersection crash).
- **Notebooks become scripts** (`scripts/01…04_*.py`) because of the time limit. A thin notebook is optional at the end.

## Layout and interface contract (every agent codes to this)
```
~/hackathon/                   # git repo -> github.com/YuvrajGupta1808/CivicTech (public)
  .venv/  (gitignored)         # python3 -m venv --system-site-packages; + geopandas statsmodels matplotlib pyyaml ridescore(git)
  AGENTS.md  .gitignore  LICENSE (MIT code; data outputs ODbL, noted in README)
  docs/plan.md  docs/progress.md  docs/memory.md  docs/architecture.md   # AVTCA-style 4 docs; plan.md = this plan
  (project files at repo root — the ordinal-crash-risk/ level below is flattened away)
    config.yaml  README.md  slides.md
    src/__init__.py  src/config.py            # load_config(path="config.yaml") -> dict (paths expanded)
    src/data/load_snapshot.py   # load_snapshot(cfg) -> GeoDataFrame; keys lower-cased; adds seg_length_m (EPSG:26985)
    src/data/fetch_crashes.py   # fetch_crashes(cfg, refresh=False) -> DataFrame (cached parquet); adds subblockkey, blockkey (norm), located, injured, near_int
    src/labels.py               # build_labels(sub_keys, crashes, cfg) -> DF[dc_subblockkey, crash_count, injury_count, midblock_count, level, level_injury, level_midblock]; join_report(...) -> dict
    src/features/ddot.py        # ddot_features(sub: DF one row per sub-block, dc_* cols) -> DF index dc_subblockkey, cols ddot_*
    src/features/osm.py         # osm_features(segments: GDF matched segments) -> DF index dc_subblockkey, cols osmf_*
    src/features/network.py     # network_features(all_segments: GDF) -> DF index dc_subblockkey, cols net_*
    src/features/groups.py      # FEATURE_GROUPS {speed, lanes_width, parking, bike, traffic, class, conflicts, calming, pavement, sidewalk, network}; SOURCES {ddot, osm, net}
    src/features/build.py       # build_table(cfg) -> DF: dc_subblockkey, dc_blockkey, ward(int), weak_match, length_m, features, labels; drops >99%-null/constant cols
    src/split.py                # ward_folds(ward: array, n_val=2) -> list[Fold(test_ward, val_wards, train_idx, val_idx, test_idx)]
    src/decode.py               # port of calibration.py: expected_level(P), fit_thresholds(mu,y,objective=macro_f1), decode(mu,thr), fit_logit_bias, argmax
    src/evaluate.py             # metrics(y, yhat, score, crash_count, length_m) -> dict (Section 14 of doc); summarise(per_fold) -> mean±sd
    src/models/baselines.py     # m0_proba(n), m1_lts(table) -> (proba, score)
    src/models/spf.py           # fit_spf(train) -> model; spf_proba(model, df) -> P(0),P(1),P(>=2) via NB pmf
    src/models/gbm.py           # fit_gbm(X,y,cfg) -> list of fitted; gbm_proba(models, X) -> mean proba
    src/models/fusion.py        # fuse({name: P}, weights) -> P; weight grid for E9
    src/models/xattn.py         # M4 (stretch): tokeniser + OSM<->DDOT cross-attn + concat variant; fit/predict per fold
    scripts/05_xattn.py         # M4 CV over the same folds -> output/oof_xattn.parquet (+ E7 source ablation, E8 concat)
    src/experiments.py          # run_cv(table, features, label_col, models=[...]) -> per-fold metrics + OOF preds (all decoders)
    scripts/01_build_dataset.py  02_ladder.py  03_ablations.py  04_findings.py
    tests/test_labels.py  test_osm_parsing.py  test_decode.py  test_split.py
    output/                     # csv/png/parquet results (hand-in parquet kept, caches not)
~/ridescore-data/  snapshot parquet, crashes_bike_2021-09-30_2026-09-29.parquet, subblock_table.parquet
```
- The join key is `dc_subblockkey` everywhere. Normalise it with `.str.strip().str.lower()`.
- Feature prefixes: `ddot_`, `osmf_`, `net_`. Ward ID, coordinates and anything derived from crashes never become features.

## Execution waves (times in EDT)
**Wave 0, Claude, 14:05–14:15**
- Install the `gh` binary into `~/bin` from the GitHub release tarball. **The user runs `~/bin/gh auth login`.** Then `git init` in `~/hackathon` and add `.gitignore`: `.venv/`, `__pycache__/`, `*.npz`, `output/cache/`, `~/ridescore-data` is outside the repo anyway.
- Create the venv and install the missing packages.
- Download the snapshot with curl from the HF `resolve` URL.
- Write the config, `load_snapshot`, `fetch_crashes` and the package skeleton, then fetch and cache the crashes.
- Copy this plan to `docs/plan.md` and write `docs/progress.md`.

**Wave 1, 5 Sonnet agents in parallel, 14:15–14:50.** Each agent owns only its files, does no `pip install`, tests against the real data in `~/ridescore-data`, and returns a summary of 15 lines or fewer.

| Agent | Owns | Done when |
|---|---|---|
| A labels | `src/labels.py`, `tests/test_labels.py` | Label counts reproduce 17,826 / 1,203 / 525 (±1%); the join report counts unmatched and sentinel crashes |
| B ddot+net | `features/ddot.py`, `features/network.py`, `features/groups.py` (ddot and net entries) | Both return one row per sub-block. Bike-best ranking, speed from OB, log AADT, truck share, width per lane. Net: legs, junctions per 100 m, endpoint arterial |
| C osm | `features/osm.py`, `tests/test_osm_parsing.py` | Parses maxspeed and lanes; normalises cycleways (crossing → none, separate → track); length-weighted aggregation excludes cycleway, path and footway segments; `osmf_has_parallel_track`. Adds the osm entries to `groups.py` (handed to B or Claude to merge) |
| D modelling | `split.py`, `decode.py`, `evaluate.py`, `models/*`, `experiments.py`, `tests/test_decode.py`, `tests/test_split.py` | Runs end to end on a synthetic table that follows the contract. Threshold grid is vectorised (coarse 0.05, then refine 0.01). SPF falls back to Poisson if NB does not converge. M1 uses the installed `ridescore` LTS |
| E writer | `README.md` skeleton, `slides.md` skeleton, `AGENTS.md`, `docs/architecture.md` (Mermaid pipeline), `docs/memory.md` | Inputs, licences (ODbL and CC BY 4.0), limitations (doc §23 plus the AADT-2020 issue and the W7/W8 join gap) and claims discipline are written; numbers are left as placeholders |
| H xattn (stretch) | `src/models/xattn.py`, `scripts/05_xattn.py` | Trains on the same folds (`split.ward_folds`) on the GPU; the env is `/home/922933190/.conda/envs/avtca/bin/python` if it has pandas, otherwise the CPU venv. Writes OOF probabilities, E7 (zero the OSM or DDOT stream) and E8 (concat). Uses a synthetic table until `subblock_table.parquet` exists. **Hard stop at 15:15.** |

Meanwhile, Claude writes `features/build.py` and `scripts/01_build_dataset.py`, then integrates each agent's work as it lands.

**Integration, Claude, 14:50–15:10**
- Build `subblock_table.parquet`.
- `scripts/02_ladder.py`: E1 (M0 → M1 → M2 → M3 → F), E3 (argmax vs logit bias vs thresholds for M2, M3 and F) and E9 (fixed vs equal vs validation-selected weights).
- Check the results make sense against M0: about 91% accuracy and macro-F1 of about 0.32.
- **About 15:05, first push:** `gh repo create YuvrajGupta1808/CivicTech --public --source . --push`, so there is something to submit even if later steps slip.

**Wave 2, 2 Sonnet agents, 15:05–15:35**
- **F ablations** (`scripts/03_ablations.py`):
  - E5: drop one feature group at a time from M3.
  - E6: DDOT+net vs OSM+net vs all.
  - E10: label variants injury-only, mid-block-only and block-level.
  - E11: drop weak matches.
  - Outputs: CSVs plus `ablation.png`.
- **G findings** (`scripts/04_findings.py`):
  - OOF surprise groups ("looks safe, but has crashes" = y=2 with predicted 0 or a bottom-half score; "looks risky, but no crashes" = predicted 2 with y=0).
  - Breakdowns by intersection share, `net_max_legs`, class, bike facility, ward and match quality.
  - `risk_map.png` (green to red) and `surprise_map.png`.
  - Export `output/crash_risk_by_segment.parquet` with the columns `osm_u, osm_v, osm_key, snapshot_date, dc_subblockkey, crash_risk_level, crash_risk_score, crash_count_5yr, scored`. Attribution goes in the Parquet metadata.

**Wrap-up, Claude, 15:35–15:55**
- Fill the README and slides with real numbers (wording: "where crashes happen", never "danger per ride").
- Final checks; update `docs/progress.md`, `docs/memory.md` and the memory files.
- If M4 has landed by 15:15, refit F with 0.4/0.4/0.2 and add E7/E8 to the README.
- About 15:55, final commit and push to CivicTech (commits end with the Claude co-author line). The user submits the repo link at 16:00.

**Cut lines**
- 14:50: if C (OSM) isn't done, build the table without `osmf_*`, and E6 becomes DDOT vs DDOT+net.
- 15:00: if the SPF breaks, F = M3 and M2 is dropped.
- 15:20: if E5 isn't done, keep only E6 plus permutation importance.
- 15:40: hard stop on analysis; write-up only.

## Verification
- `cd ~/hackathon && .venv/bin/python -m pytest -q`: all 4 test files pass.
- `gh repo view YuvrajGupta1808/CivicTech` shows the repo as public, the latest commit is pushed, and no raw data or `.venv` is committed.
- `01_build_dataset.py` prints 19,554 sub-blocks, the label counts above, the crash match rate (88.8%) and the sentinel count (142).
- `02_ladder.py`:
  - M0 macro-F1 is about 0.32.
  - Every model's thresholds come from validation wards only. Assert in `split.py` that no ward plays two roles in a fold.
  - Results are reported as mean±sd over 8 folds plus a table per ward.
- Hand-in Parquet:
  - 28,978 rows, unique `(osm_u, osm_v, osm_key)`.
  - `scored == False` exactly for the 1,399 rows with `match_status == none`.
  - Levels are in {0, 1, 2} and scores in [0, 2].
- Open `risk_map.png` and `surprise_map.png` and check by eye that they cover DC with no blank wards.

## Outcome (2026-10-03, 14:30 EDT)

- All waves finished by 14:30, ahead of the cut lines; M4 landed and is in the fusion.
- E1: F (0.4 M4 + 0.4 M3 + 0.2 M2) macro-F1 0.456 ± 0.029, AUC 0.769, top-10% capture 0.435; M0 0.315; M1 LTS v1 0.283.
- E3: thresholds beat argmax for SPF (+7.9 pts), GBM (+2.1) and F (+4.6).
- E5/E6: network geometry is the strongest group; DDOT + net ≈ all; OSM + net is weaker on top-10% capture.
- E7/E8: blanking DDOT costs more than blanking OSM; cross-attention ≈ concatenation.
- H5 rejected: "looks safe, has crashes" sub-blocks are mostly mid-block.
- Repo: https://github.com/YuvrajGupta1808/CivicTech (public). PR to ridescoredc-models was dropped by the user's decision.
