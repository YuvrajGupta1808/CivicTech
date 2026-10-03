# Memory: key decisions and non-obvious constraints

## Facts verified against the live data

- Crashes layer is MapServer/24; window `REPORTDATE >= DATE '2021-09-30' AND REPORTDATE < DATE '2026-09-30'` gives 3,089 bike crashes; pages hold 1,000 rows, paginate with `resultOffset`.
- Crash `SUBBLOCKKEY` is 32-character lowercase hex; 142 crashes carry the sentinel `'Route not found'` instead and count as unlocated.
- `OFFINTERSECTION` is in metres, not feet.
- The snapshot has 28,978 segments that collapse to 19,554 DDOT sub-blocks; `dc_*` values are constant within a sub-block, so one row per sub-block is safe.
- 88.8% of crashes match a `dc_subblockkey` (93.0% excluding the sentinel); the remaining unmatched crashes sit on sub-blocks outside the snapshot.
- Wards 7 and 8 match only about 77% of crashes (Ward 1 about 96%): an equity caveat that goes in the limitations.
- Label counts: 0 = 17,826 (91.2%), 1 = 1,203 (6.2%), 2+ = 525 (2.7%). Level-2 by ward: W1 104, W2 185, W3 12, W4 37, W5 68, W6 87, W7 11, W8 21.
- `dc_SPEEDLIMITS_IB` is 100% null; use `dc_SPEEDLIMITS_OB` and fall back to `_OB_ALT`.
- Always-null columns to drop: `dc_LEFTTURN_EXCLUSIVE`, `dc_RIGHTTURN_EXCLUSIVE`, `dc_*CURBLANE_EXCL`, `dc_MIDMEASURE`.
- `dc_AADT` is present on 42% of matched rows and its year is 2020 for every one, so `aadt_is_2020` is dropped and the pandemic-year issue goes in the limitations.
- `osm_cycleway = "crossing"` (1,513 rows) is a crossing, not a lane: map it to `none`. `osm_highway = cycleway` (3,134 rows) is a separately mapped track.
- 1,399 segments (4.8%) have `match_status == none`; they are unscored in the hand-in.
- Hand-in row key follows PR #14 of ridescoredc-models: `osm_u, osm_v, osm_key` plus `snapshot_date = "2026-09-29"`.

## Design decisions

- No PR to ridescoredc-models; this is a standalone public repo (YuvrajGupta1808/CivicTech). `ridescore` is installed from the `develop` branch.
- Unit is the DDOT sub-block (not the block, not the OSM edge); block-level is only label variant E10.
- Split is leave-one-ward-out: test = ward w, validation = next 2 wards cyclically, train = other 5, so every fold has at least 32 level-2 validation sub-blocks (Wards 3 and 7 have about 11 each).
- Each sub-block gets one out-of-fold prediction from the fold where its ward is the test ward; maps, surprise analysis and the hand-in use only these.
- Thresholds, logit biases and fusion weights are fitted on validation wards only; `ward_folds` asserts no ward plays two roles.
- Threshold decoding objective is macro-F1 (not accuracy as in AVB-Engage `_score_candidate`); grid is vectorised, coarse 0.05 then refine 0.01.
- M1 is the production LTS v1 (`ridescore.models.ridescore_v1.lts.lts_level`); LTS 1-2 maps to 0, 3 to 1, 4 to 2, score = LTS level. This covers E2. If pip install fails, vendor `lts.py` and `normalise.py` with attribution.
- M2 is a negative binomial SPF with offset log(length), falling back to Poisson if NB does not converge.
- M3 follows the AVB-Engage GBM recipe with sample weights 1/sqrt(class count), `early_stopping=False`, native missing values (no blanket `*_missing` columns), categoricals as pandas `category` with `categorical_features="from_dtype"`.
- Fusion F = 0.67 M3 + 0.33 M2, fixed weights; with M4: 0.4 M4 + 0.4 M3 + 0.2 M2.
- M4 (OSM-DDOT cross-attention, d=32, 4 heads, 2 exchanges, source dropout 0.15, ordinal loss lambda 0.15, label smoothing 0.1, 3 seeds) is a stretch track with a hard stop at 15:15; if it misses, the GBM ladder is the result.
- Length is `dc_LENGTH` in metres, falling back to summed road-segment length in EPSG:26985.
- Intersection cutoff for label variant B is 15 m (`OFFINTERSECTION <= 15`).
- Ward id, coordinates and anything crash-derived never become features; feature prefixes are `ddot_`, `osmf_`, `net_`.
- Notebooks became numbered scripts (`scripts/01` to `05`) because of the time limit.
- Wording rule: "where crashes happen", never "danger per ride"; no exposure data exists.
- Data outputs are ODbL (OSM-derived); code is MIT; DDOT and Crashes in DC are CC BY 4.0.
- Sanity check: M0 gives 0.900 accuracy and 0.315 macro-F1 (fold mean); anything far from that means a label or split bug.
- pandas 3 + sklearn 1.8: HistGradientBoosting with `categorical_features="from_dtype"` fails on `str`-dtype categories containing NA; `experiments._clean_features` recodes every categorical to integer categories.
- GBM threads are capped at 8 (`threadpool_limits`); under CPU contention an uncapped fold took 400 s instead of 6 s.
- M4 runs in the separate GPU env `/home/922933190/.conda/envs/avtca/bin/python` (torch + CUDA, no PyYAML), so its config values are hardcoded in `scripts/05_xattn.py`; it writes `output/xattn_probs.parquet`, which `02_ladder.py` picks up as an external model.
- M4's best validation epoch is usually 3-8 of 40, so it overfits early; cross-attention did not beat concatenation (E8).
- Production LTS v1 is computed once in `build.py` as `ref_lts` (a reference column, never a feature) using the `ridescore` package installed from git.
- `OFFINTERSECTION` is in metres (checked against route measures); 15 m is the intersection cutoff for label variant B.
- Ablations ran with `--fast` (first GBM fit only), so their "all features" base is 0.444, not the ladder's 3-fit 0.439.
- BNA is computed with the team's `bikescore-bna` 0.2.0 (git 6a6cb494) segment-stress rules only (`default_segment_stress_rules()`); intersection stress needs the node graph and is not used. Inputs are DDOT first, then OSM, then BNA per-class defaults.
- Binary scores (BNA, LTS-coarse) are compared with AUC and top-10% capture only; macro-F1 is not comparable to the 3-class models.
- The LTS disagreement cells use LTS ≤ 2 vs ≥ 3 and fusion out-of-fold level 0 vs ≥ 1; in both disagreement cells the crash record sides with the model.
- Fairness check (scripts/10): GBM on LTS's 4 raw facts reaches AUC 0.706 / top-10% 36% vs LTS as-is 0.623 / 18%; re-weighting only the LTS level gives 0.642 / 20%. Most of the gain is from crash-weighted combination of the same facts.
