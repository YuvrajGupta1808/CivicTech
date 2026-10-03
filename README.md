# CivicTech: Do street-design scores explain where cyclists crash in DC?

RideScore DC Challenge 1 (bike safety scores), ideas 4.2 and 4.3. Civic Tech DC x GW OSPO hackathon.

**Answer:** {{HEADLINE}}

---

## Why

RideScore DC colours every street by its design: speed limit, lanes, bike facility, traffic and the Level of Traffic Stress (LTS) that follows from them. Those colours describe how a street is built. Nobody has checked them against what has happened on the street. This repo does that check with five years of police-reported bike crashes, and it measures how much of the pattern of crash locations the design attributes can explain at all.

The scores are not a danger-per-ride measure. We have no exposure data (no counts of how many people ride where), so every result is about where recorded crashes happen.

## Question and hypotheses

**Question:** how far do street-design attributes (OpenStreetMap plus DDOT Roadway SubBlock, via the RideScore DC basemap snapshot of 2026-09-29) explain where cyclist crashes happen in DC?

| ID | Hypothesis | Tested by |
|---|---|---|
| H1 | Design attributes carry signal about crash locations: a learned model beats the majority-class reference on held-out wards. | E1 |
| H2 | Learned models (SPF, gradient boosting, fusion) rank sub-blocks by recorded crashes better than production LTS v1. | E1, E9 |
| H3 | Speed, lanes and width, traffic and road class carry most of the signal; bike facilities carry less. | E5 |
| H4 | DDOT and OSM overlap but are not interchangeable; using both beats either alone. | E6 (E7, E8 if M4 lands) |
| H5 | Thresholds fitted on validation wards beat plain argmax at finding the rare 2+ class. | E3 |

These are hypotheses, not findings. Section Results says which held.

## Data

| Source | Used for | Licence | Link |
|---|---|---|---|
| RideScore DC basemap snapshot, 2026-09-29 (`ridescore_dc_basemap_2026-09-29.parquet`, 28,978 road segments, OSM tags joined to DDOT Roadway SubBlock attributes) | Features, geometry, the join key `dc_subblockkey` | ODbL, "© OpenStreetMap contributors"; DDOT part CC BY 4.0 | Hugging Face file named in the hackathon brief; project: https://github.com/civictechdc/ridescoredc-models |
| DDOT Roadway SubBlock (carried inside the snapshot as `dc_*` columns) | Speed limit, lanes, width, AADT, parking, bike facility, pavement, sidewalk | CC BY 4.0, Government of the District of Columbia / DDOT | https://opendata.dc.gov/ |
| Crashes in DC (MPD / DDOT), layer 24 of `Public_Safety_WebMercator` | Labels only | CC BY 4.0, Government of the District of Columbia / MPD / DDOT | https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Public_Safety_WebMercator/MapServer/24 |

Crashes are never used as features. They give the label and nothing else.

## Method

| Step | Choice |
|---|---|
| Unit | DDOT sub-block. The 28,978 snapshot segments collapse to **19,554 sub-blocks**; `dc_*` values are constant inside one. |
| Label | Bike-involved crashes (`TOTAL_BICYCLES > 0`) per sub-block, 2021-09-30 to 2026-09-29 (3,089 crashes), binned **0 / 1 / 2+**: 17,826 / 1,203 / 525 sub-blocks. |
| Features | Eleven groups from three sources: DDOT (`ddot_*`), OSM (`osmf_*`), network shape (`net_*`). Groups: speed, lanes_width, parking, bike, traffic, class, conflicts, calming, pavement, sidewalk, network. Ward, coordinates and anything crash-derived are excluded. |
| Split | Leave-one-ward-out. Test = ward w, validation = the next two wards in cyclic order, train = the other five. Every sub-block gets one out-of-fold (OOF) prediction. |
| Reference | M0, the majority class. Every number is read against it. |
| Decoding | Expected level `mu = sum_k k * P(k)`; two thresholds fitted on validation wards to maximise macro-F1 (coarse 0.05 grid, then 0.01). Compared with argmax and a logit bias. |
| Imbalance | Inverse-sqrt class weights for the gradient boosting model. |
| Fusion | Fixed-weight late fusion of probabilities; weights from `config.yaml`, not tuned on test wards. |

Model ladder:

| Model | What it is |
|---|---|
| M0 | Majority class (always 0) |
| M1 | Production RideScore LTS v1 (`ridescore.models.ridescore_v1.lts`). LTS 1-2 maps to 0, LTS 3 to 1, LTS 4 to 2; the score is the LTS level. This is also experiment E2. |
| M2 | Safety Performance Function: negative binomial regression of crash count with offset log(length) (Poisson if NB does not converge) |
| M3 | Gradient boosting (histogram GBM, native missing values, 3 fits averaged) |
| F | Late fusion, 0.67 M3 + 0.33 M2 |
| M4 (stretch) | OSM-to-DDOT cross-attention Transformer. Reported only if it finished in time: {{M4_STATUS}}. If it lands, F becomes 0.4 M4 + 0.4 M3 + 0.2 M2. |

Method lineage: AVT-CA (audio-visual cross-attention) -> AVB-Engage (our ordinal engagement paper) -> this project, which reuses the ordinal toolkit (threshold decoding fitted on validation, class weighting, fixed-weight late fusion, ward-independent CV, majority-class reference) on a different problem.

Experiments: E1 ladder, E3 decoders, E5 feature-group ablation, E6 DDOT vs OSM, E9 fusion weights, E10 label variants (injury-only, mid-block-only, block-level), E11 drop weak OSM-to-DDOT matches; E7 and E8 (source ablation, concat variant) only if M4 lands.

## Results

All numbers are mean +- sd over the 8 leave-one-ward-out folds unless stated. Placeholders are filled by the final run.

**E1, ladder (M0 to F):**

{{E1_TABLE}}

**E3, decoders (argmax vs logit bias vs thresholds):**

{{E3_TABLE}}

**E5, feature-group ablation (drop one group from M3):**

{{E5_FIG}}

**E6, DDOT vs OSM vs both (each with network features):**

{{E6_TABLE}}

**E7 / E8, source ablation and concat variant (only if M4 landed):**

{{E7_E8_TABLE}}

**E9, fusion weights (fixed vs equal vs validation-selected):**

{{E9_TABLE}}

**E10 / E11, label variants and weak matches:**

{{E10_TABLE}}

{{E11_TABLE}}

**Per ward:**

{{PER_WARD_TABLE}}

**Risk map (OOF expected level, green to red):**

{{RISK_MAP}}

**Surprise map.** "Looks safe but has crashes" is a sub-block with 2+ crashes that the model placed in level 0 or the bottom half of scores. "Looks risky but no crashes" is the reverse. These are disagreements between the model and the record, not verdicts on a street. Counts and breakdowns by intersection share, `net_max_legs`, class, bike facility, ward and match quality:

{{SURPRISE_MAP}}

## Hand-in file

`output/crash_risk_by_segment.parquet`: one row per snapshot segment (28,978 rows, unique `(osm_u, osm_v, osm_key)`), keyed like the other RideScore hand-ins.

| Column | Meaning |
|---|---|
| `osm_u`, `osm_v`, `osm_key` | OSM edge key |
| `snapshot_date` | `2026-09-29` |
| `dc_subblockkey` | DDOT sub-block the segment belongs to (null if unmatched) |
| `crash_risk_level` | Decoded level in {0, 1, 2}; from the out-of-fold prediction for the sub-block's ward |
| `crash_risk_score` | Expected level in [0, 2]. Higher means more recorded crashes expected. Not a probability and not per-ride risk. |
| `crash_count_5yr` | Bike-involved crashes matched to the sub-block, 2021-09-30 to 2026-09-29 |
| `scored` | False for the 1,399 segments (4.8%) with no sub-block match; their level and score are empty |

Attribution (OSM, DDOT, MPD) is stored in the Parquet metadata.

## How to reproduce

```bash
python3 -m venv .venv --system-site-packages     # reuses pandas, sklearn, pyarrow, torch, pytest from the base env
.venv/bin/pip install geopandas statsmodels matplotlib pyyaml \
  "ridescore @ git+https://github.com/civictechdc/ridescoredc-models@develop"

# put the snapshot at ~/ridescore-data/ridescore_dc_basemap_2026-09-29.parquet (paths: config.yaml)

.venv/bin/python scripts/01_build_dataset.py   # fetch + cache crashes, build labels and the sub-block table
.venv/bin/python scripts/02_ladder.py          # E1, E3, E9: M0 -> M1 -> M2 -> M3 -> F
.venv/bin/python scripts/03_ablations.py       # E5, E6, E10, E11
.venv/bin/python scripts/04_findings.py        # surprise analysis, maps, hand-in Parquet
.venv/bin/python scripts/05_xattn.py           # optional: M4, E7, E8 (GPU helps)

.venv/bin/python -m pytest -q
```

`01_build_dataset.py` prints the sanity numbers to check against: 19,554 sub-blocks; label counts 17,826 / 1,203 / 525; crash match rate 88.8%; 142 sentinel ("Route not found") crashes. M0 should score about 91% accuracy and about 0.32 macro-F1.

## Limitations

- **No exposure.** There are no ridership or bike-count data, so busy streets can look risky just because more people ride them. Results are about recorded crashes, not risk per ride.
- **Associations, not causes.** A feature that predicts crashes is not shown to cause them.
- **Rare events.** Only 525 sub-blocks are level 2+. Ward 3 has 12 and Ward 7 has 11, so per-ward scores are noisy.
- **Police-reported only.** Crashes that were not reported, and near misses, are absent.
- **Crash-to-street assignment.** Intersection crashes are assigned to a sub-block by the source data; label variants (E10) test how much this matters.
- **AADT is a pandemic year.** Where `dc_AADT` exists (42% of matched rows) its year is 2020 in every case.
- **Pavement data covers 2019-2023.**
- **Unmatched crashes.** 11% of crashes (347 of 3,089) do not match a snapshot sub-block: 142 are "Route not found" and the rest sit on sub-blocks outside the snapshot.
- **Equity caveat.** The loss is uneven. Wards 7 and 8 lose about 23% of their crashes against 4% in Ward 1, so labels there are more likely to be missing and the model is trained on less of the truth.
- **Unscored segments.** 4.8% of snapshot segments (1,399) have no sub-block match and are not scored.
- **Trend.** Bike crashes rose from 2022 to 2025; the five-year count pools different years.
- **Not a replacement for LTS.** This is a check on design scores, not a new safety rating.

## Claims discipline

- Say "where crashes happen" or "where recorded crashes are expected". Never "danger per ride", "unsafe street" or "safe street".
- Every result is read against M0.
- Surprise maps show model-record disagreement, not blame.
- No causal language.
- Ward, coordinates and crash-derived values are never features; thresholds and weights are fitted on validation wards only.

## Credits and licence

- Code: MIT (see `LICENSE`).
- Data outputs (`output/`, including the hand-in Parquet): ODbL, because they derive from OpenStreetMap. "© OpenStreetMap contributors". DDOT and Crashes in DC data: CC BY 4.0, Government of the District of Columbia / DDOT / MPD.
- RideScore DC: https://github.com/civictechdc/ridescoredc-models (production LTS v1 used as M1).
- Method lineage: AVT-CA, AVB-Engage.
- Author: Yuvraj Gupta, for the Civic Tech DC x GW OSPO hackathon.
