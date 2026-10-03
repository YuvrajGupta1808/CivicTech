# Demo script: "Does street design explain where DC cyclists crash?"

Two slides in the shared deck (Models section). Two minutes, one presenter. Lead with one point: **RideScore's comfort scores don't show where cyclists crash; a model built on the same street data finds about 3 times more crash locations.**

---

## Part A: what to say (about 2 minutes)

### Slide 1 (about 60 s)

> RideScore colours every DC street green to red from its design, using LTS and BNA. Nobody had checked those colours against where cyclists actually crash. So we asked: does street design explain where DC cyclists crash?
>
> We had five hypotheses, on the left. The main one: design predicts crash locations, but only modestly.
>
> Here's what we built, left to right. We took the organisers' street snapshot, which joins OpenStreetMap and DDOT city records, plus 3,089 police-reported bike crashes from Open Data DC. We joined them by DDOT block ID into one table: 19,554 blocks, 57 design features, and how many crashes each block had in five years.
>
> Three models learn from that table. One is the cross-attention model from our AVB-Engage research, where OpenStreetMap and DDOT read each other the way audio and video do in emotion recognition. We combine the three models, and we test only on wards the model never saw.

### Slide 2 (about 60 s)

> Here's the result. Take the 10% of blocks each score rates worst, in a ward the model never saw. Our model's 10% holds 44% of the bike crashes. LTS's holds 13%, BNA's 18%, and random gives 10%.
>
> That doesn't mean LTS is wrong. It measures comfort, and comfort isn't crash risk. LTS's calmest streets have the most crashes per kilometre, like 14th Street at Irving, with a protected lane and 18 crashes in five years, because that's where people ride.
>
> Four of our five hypotheses held. One surprise: streets that look safe but have crashes are mostly mid-block, not at intersections.
>
> We have scores for every street segment, maps, and code on GitHub. Next: show a crash layer next to LTS, add bike counts so we can separate busy from dangerous, and fix the Ward 7 and 8 crash-matching gap. Honest limit: without ridership data this shows where crashes happen, not how dangerous one ride is. Thank you.

---

## Part B: slide 1 explained, piece by piece

### The diagram ("What we built"), left to right

| Stage | What it is | Plain words |
|---|---|---|
| **Public data** | OpenStreetMap; DDOT SubBlock; Crashes in DC | OSM is the volunteer map, good on bike lanes and one-way streets. DDOT is the city's official records, good on lanes, speed limits, traffic counts and pavement. Crashes are 3,089 police-reported bike crashes, 30 Sep 2021 to 29 Sep 2026. |
| **One table** | 19,554 sub-blocks × 57 features + label | One row per DDOT block (a short stretch of street, about 90 m). Crashes join by the block ID they already carry. Features: 36 from DDOT, 16 from OSM, 7 about layout (how many roads meet at each end, junctions per 100 m, length). Label: 0, 1 or 2+ crashes in 5 years. |
| **M2 SPF** | Safety Performance Function (negative binomial) | The standard traffic-engineering formula: crashes per metre go up or down with traffic, lanes, speed and road type. Easy to read. |
| **M3 Gradient boosting** | Hundreds of small decision trees | Rules like "4+ lanes and a 4-way junction means higher risk", each fixing the last one's mistakes. Rare crash blocks get extra weight. |
| **M4 Cross-attention** | Transformer from AVT-CA / AVB-Engage | Every fact becomes a token. OSM tokens and DDOT tokens form two groups that read each other, so the model learns which source to trust when they disagree. 15% of the time in training one whole source is hidden, so the model can't lean on just one. |
| **Late fusion** | 0.4 M4 + 0.4 M3 + 0.2 M2 | A weighted vote of three experts. The weights were fixed in advance, not tuned on test data. |
| **Ordinal decoding** | Thresholds fitted on validation wards | Turn the three chances (0, 1, 2+) into one score from 0 to 2, then pick two cut-offs on practice wards. Crashes are rare, so "pick the most likely" would almost always say 0. |
| **Outputs** | Risk per segment; maps; checks | A level and score for all 28,978 street segments, risk and disagreement maps, and comparisons with "always no crash", LTS and BNA. |
| **Dashed band** | Leave-one-ward-out | Train on 5 wards, set the cut-offs on 2, test on 1, and repeat for all 8 wards. Every score comes from a model that never saw that ward. |

### The four number tiles

- **3,089:** bike-involved crashes in the 5-year window. 89% matched a block; the rest are reported, not hidden.
- **19,554:** DDOT street blocks, one table row each.
- **57:** street-design facts per block. No location, ward or crash information is ever used as an input.
- **8 wards:** each one is tested on a model that never saw it.

### The five hypotheses, and why we expected them

| ID | Hypothesis | Why we expected it |
|---|---|---|
| H1 | Design predicts crash locations, but only modestly | Crashes also depend on how many people ride, on behaviour and on luck, none of which is in street data. |
| H2 | Junctions, lanes and traffic matter more than bike-lane type | Most serious conflicts involve turning cars and busy roads. A painted lane alone may not change that. |
| H3 | DDOT is more useful than OSM | DDOT has speed limits for 84% of streets and lanes for 100%; OSM has 30% and 48%. |
| H4 | Ordinal decoding helps find rare repeat-crash blocks | In AVB-Engage, fitted thresholds beat "pick the most likely" on imbalanced data. Here 91% of blocks have no crash. |
| H5 | Streets that look safe but crash are mostly at intersections | Intersections are classic conflict points that block-level design data describes poorly. |

---

## Part C: slide 2 explained, piece by piece

### The bars (headline result)

**Top-10% capture:** in a ward the model never saw, take the 10% of blocks a score rates worst and count what share of that ward's real bike crashes happened on them.

| Score | Share of crashes caught |
|---|---|
| Random | 10% |
| LTS (RideScore today) | 13% |
| BNA (team's second model) | 18% |
| **Our model** | **44%** |

### Hypotheses: what held, and the evidence behind each line

| ID | Verdict | Evidence |
|---|---|---|
| H1 | Supported | 44% vs 10%. AUC 0.77, where 0.5 is a coin flip and 1 is perfect. But only 32% of repeat-crash blocks are put in the top level, so the effect is clear but modest. |
| H2 | Partly | Removing junction / layout facts hurts most (AUC −0.038), then lanes and width (−0.025 macro-F1). Bike-lane type on its own is within noise. Traffic mattered less than expected, probably because traffic counts exist for only a third of blocks and come from 2020. |
| H3 | Supported | DDOT + layout scores macro-F1 0.429 and catches 41%; OSM + layout scores 0.411 and catches 36%; both together 0.444. Cross-attention (0.432) did **not** beat simply concatenating all facts (0.438). We report that honestly. |
| H4 | Supported | Fused model macro-F1 0.410 with "most likely" vs 0.456 with fitted thresholds (+4.6 points); the safety formula gains +7.9. |
| H5 | Rejected | 79% of crashes on "looks safe, has crashes" blocks happened more than 15 m from an intersection. Those blocks are mostly local and collector streets with fewer bike lanes. |

### Key evidence box

- **LTS's calmest streets have the most crashes per km:** LTS 1 has 6.5 crashes per km, against 0.85 for LTS 2. LTS 1 is mostly protected lanes and trails, which is where riders concentrate.
- **14th St NW at Irving St:** the worst block in DC, with 18 crashes in 5 years, LTS 1 and a protected lane. It's one of the busiest bike corridors.
- **The takeaway:** crashes follow riders.

### Where we are, next, limit

- **Where we are:** a risk level and score for all 28,978 segments (`output/crash_risk_by_segment.parquet`), static and interactive maps, a README, 80 passing tests, all on GitHub.
- **Next:**
  - show a crash layer next to LTS, not instead of it
  - add bike counts to separate busy from dangerous
  - fix Ward 7–8 crash matching (they lose about 20% of their crashes)
- **Limit:** no ridership data, so this shows where crashes happen, not danger per ride.

---

## Part D: likely questions (short answers)

- **"Isn't it unfair? Your model learned from crashes."**
  - Partly, yes, and that's the point: comfort rules weren't built for crashes.
  - But it never saw the test ward's crashes.
  - With only LTS's own four facts (speed, lanes, bike lane, road type) weighted by crash data, the capture goes from 18% to 36%. So most of the gap is "built for comfort vs fitted to crashes".
- **"So are protected bike lanes dangerous?"**
  - No. They carry the most riders. Without bike counts we can't separate busy from dangerous.
- **"Why does cross-attention matter if it didn't win?"**
  - It gave the best single-model ranking (AUC 0.770) and lifts the fused model.
  - The extra cross-attention block itself didn't beat simple concatenation; we report that.
- **"Within the same road type, how do LTS and BNA do?"**
  - About a coin flip (AUC 0.47–0.57); our model gets 0.67–0.73.
- **"What's macro-F1?"**
  - How well a model finds each of the three levels (0, 1, 2+), averaged so the rare levels count equally.
  - "Always no crash" gets 0.315; our fused model 0.456; LTS 0.283.
- **"Can RideScore use it?"**
  - Yes. Per-segment scores are keyed by `osm_u`, `osm_v`, `osm_key` like the other hand-ins, and there's an interactive map in `output/risk_map.html`.
