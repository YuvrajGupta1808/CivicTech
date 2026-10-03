# Slides: RideScore DC crash check

Two slides for the shared demo deck. All numbers come from `output/` (see README).

---

## Slide 1: Problem and finding

**Title:** Do street-design scores explain where cyclists crash in DC?

**Problem**
- RideScore DC colours streets by design (speed, lanes, bike facility, LTS). Nobody has checked those colours against crashes.
- Test: 19,554 DDOT sub-blocks, 3,089 bike-involved crashes (2021-09-30 to 2026-09-29), binned 0 / 1 / 2+ (17,826 / 1,203 / 525).
- Honest test: leave-one-ward-out, thresholds fitted on validation wards, every score read against "always 0".

**Finding**
- Headline: Street design explains part of where DC cyclists crash, and RideScore's LTS barely tracks it
- Macro-F1, mean over 8 held-out wards: always-0 0.315, production LTS v1 0.283, boosting 0.439, fused 0.456.
- Top 10% of sub-blocks by score hold 44% of a held-out ward's crashes (random: 10%).
- LTS check: the calmest LTS-1 streets have the most crashes per km (6.5 vs 0.85 for LTS 2), likely because riders concentrate there.
- M4 (OSM ↔ DDOT cross-attention, from AVT-CA / AVB-Engage) has the best single-model AUC (0.770); fused 0.4 M4 + 0.4 GBM + 0.2 SPF.
- Biggest feature group: network geometry (intersection legs, junction density), then lanes/width; bike-facility type alone is within noise. DDOT vs OSM: DDOT carries a bit more than OSM (macro-F1 0.429 vs 0.411); together 0.444; cross-attention gives no gain over plain concatenation.

**Figures** (left to right)
- `docs/architecture.png`: one-line pipeline (data → table → 3 models → fusion + decoding → outputs)
- `output/ladder.png`: model ladder M0 to F with sd bars (E1)
- `output/ablation.png`: drop-one feature-group ablation (E5)
- `output/risk_map.png`: out-of-fold expected level, green to red

---

## Slide 2: Where it fails and next steps

**Where it fails**
- `output/surprise_map.png`: 176 sub-blocks look safe by design but have 2+ crashes; 330 look risky but have none.
- Surprises cluster by: mid-block crashes (79%) on local and collector streets with fewer bike facilities, not intersections (H5 rejected); e.g. 1st St NW at K St (9 crashes), Eastern Ave NE (8).
- No exposure data: busy streets can look risky because more people ride them.
- 11% of crashes do not match a sub-block; Wards 7-8 lose about a fifth against 3% in Ward 1 (equity caveat).
- Only 525 level-2 sub-blocks; Wards 3 and 7 have about 11 each.
- AADT is a 2020 pandemic-year value; 4.8% of segments are unscored.

**LTS, BNA and our model**
- Fairness check: with only LTS's own 4 facts, but weighted by crash data, the top-10% capture doubles (18% → 36%). The gap is mostly "built for comfort vs fitted to crashes"; junction layout adds the rest (→ 41–44%).
- BNA (team package) vs LTS agree on 81% of blocks. As crash-location signals: BNA AUC 0.65, LTS 0.60, our model 0.77; crashes/km on "uncomfortable" streets: BNA 2.9, LTS 2.1, our high level 5.2 (`output/bna_compare.png`).
- Where they disagree, the crash record sides with the model: "LTS calm, model high" blocks have 5.3 crashes/km (same as both-high, 5.2); "LTS stressful, model low" blocks have 1.1 (close to both-low, 0.7). Map: `output/lts_disagreement_map.png`; interactive: `output/risk_map.html`.

**Next steps**
- Add exposure: bike counts, bikeshare trips, ridership estimates.
- Fix the Wards 7-8 crash-to-street gap.
- Refresh AADT and pavement; test across time, not only across wards.
- Feed the OOF level and score into RideScore as a validation layer, not a replacement for LTS.

**Figures**
- `output/surprise_map.png` (main), `output/risk_map.png` (inset)

**Claim:** where recorded crashes happen, never danger per ride.

---

## Speaker script (about 2 minutes, about 250 words)

RideScore DC colours every street by how it is built: speed limit, lanes, bike facility, traffic stress. The colours are a design judgement. We asked a simple question: how far do those design attributes explain where cyclists actually crash?

We took 19,554 DDOT sub-blocks and counted police-reported bike crashes on each over five years, 3,089 crashes in all. Most blocks have none, so we binned them into zero, one, and two or more. To keep the test honest we held out one ward at a time, fitted every threshold on other wards, and compared everything with a model that always says zero. That model is right 91% of the time, which is why accuracy alone means nothing here.

The result: Street design explains part of where DC cyclists crash, and RideScore's LTS barely tracks it. The production LTS score reaches a macro-F1 of 0.283. Gradient boosting on the same design data reaches 0.439, and the fused model 0.456, against 0.315 for always-zero. The feature group that matters most is network geometry (intersection legs, junction density), then lanes/width; bike-facility type alone is within noise.

Now the limits. The map shows where the model and the crash record disagree: 176 blocks look safe but have repeated crashes. We have no ridership data, so this says where crashes are recorded, not danger per ride. Eleven percent of crashes cannot be placed on a block, and Wards 7 and 8 lose about a fifth of theirs, so those wards are under-counted.

Next: add exposure data, fix the Wards 7 and 8 gap, and offer our out-of-fold score to RideScore as a check on its design colours, not a replacement. Thank you.
