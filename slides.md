# Slides: RideScore DC crash check

Two slides. Numbers in `{{...}}` are filled from `output/` after the final run.

---

## Slide 1: Problem and finding

**Title:** Do street-design scores explain where cyclists crash in DC?

**Problem**
- RideScore DC colours streets by design (speed, lanes, bike facility, LTS). Nobody has checked those colours against crashes.
- Test: 19,554 DDOT sub-blocks, 3,089 bike-involved crashes (2021-09-30 to 2026-09-29), binned 0 / 1 / 2+ (17,826 / 1,203 / 525).
- Honest test: leave-one-ward-out, thresholds fitted on validation wards, every score read against "always 0".

**Finding**
- Headline: {{HEADLINE}}
- Macro-F1, mean over 8 held-out wards: always-0 {{M0_F1}}, production LTS v1 {{M1_F1}}, boosting {{M3_F1}}, fused {{F_F1}}.
- Top {{CAPTURE_K}}% of sub-blocks by score hold {{CAPTURE_TOP}}% of crashes.
- Biggest feature group: {{TOP_GROUP}}. DDOT vs OSM: {{DDOT_VS_OSM}}.

**Figures** (left to right)
- `output/ladder.png`: model ladder M0 to F with sd bars (E1)
- `output/ablation.png`: drop-one feature-group ablation (E5)
- `output/risk_map.png`: out-of-fold expected level, green to red

---

## Slide 2: Where it fails and next steps

**Where it fails**
- `output/surprise_map.png`: {{N_SURPRISE_SAFE}} sub-blocks look safe by design but have 2+ crashes; {{N_SURPRISE_RISKY}} look risky but have none.
- Surprises cluster by: {{SURPRISE_PATTERN}}.
- No exposure data: busy streets can look risky because more people ride them.
- 11% of crashes do not match a sub-block; Wards 7-8 lose about 23% against 4% in Ward 1 (equity caveat).
- Only 525 level-2 sub-blocks; Wards 3 and 7 have about 11 each.
- AADT is a 2020 pandemic-year value; 4.8% of segments are unscored.

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

The result: {{HEADLINE}}. The production LTS score reaches a macro-F1 of {{M1_F1}}. Gradient boosting on the same design data reaches {{M3_F1}}, and the fused model {{F_F1}}, against {{M0_F1}} for always-zero. The feature group that matters most is {{TOP_GROUP}}.

Now the limits. The map shows where the model and the crash record disagree: {{N_SURPRISE_SAFE}} blocks look safe but have repeated crashes. We have no ridership data, so this says where crashes are recorded, not danger per ride. Eleven percent of crashes cannot be placed on a block, and Wards 7 and 8 lose almost a quarter of theirs, so those wards are under-counted.

Next: add exposure data, fix the Wards 7 and 8 gap, and offer our out-of-fold score to RideScore as a check on its design colours, not a replacement. Thank you.
