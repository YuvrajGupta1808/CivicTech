# Progress

| Item | Owner | Status | Notes |
|---|---|---|---|
| Env, data, config, loaders, split | Claude | done | 3,089 crashes cached; 19,554 sub-blocks; 2,742 crashes matched (88.8%) |
| Labels (A) | Sonnet | done | 17,826 / 1,203 / 525; variants injury, mid-block, block |
| DDOT + network features (B) | Sonnet | done | 36 ddot_, 7 net_ |
| OSM features (C) | Sonnet | done | 16 osmf_ (osmf_width dropped, 0.3% filled) |
| Decode + evaluate (D1) | Sonnet | done | macro-F1 thresholds, logit bias, capture metrics |
| Models + CV runner (D2) | Sonnet | done | M0, M1 LTS v1, M2 SPF (NB converged), M3 GBM, fusion |
| Writer (E) | Sonnet | done | README, slides, AGENTS, LICENSE, docs |
| M4 cross-attention (H) | Sonnet | done | AUC 0.770; fused 0.4/0.4/0.2; E7, E8 |
| Ladder E1, E3, E9 | Claude | done | F macro-F1 0.456, AUC 0.769, top-10% capture 0.435 |
| Ablations E5, E6, E10, E11 (F) | Sonnet | done | network geometry matters most; DDOT > OSM slightly |
| Findings, maps, hand-in (G) | Sonnet | done | 176 looks-safe / 330 looks-risky; 28,978-row Parquet |
| README + slides filled | Claude | done | no placeholders left |
| Push to github.com/YuvrajGupta1808/CivicTech | Claude | done | public |
| BNA comparison (bikescore-bna 0.2.0) | Sonnet | done | agrees with LTS on 81%; AUC 0.647 vs LTS 0.599 vs fusion 0.769 |
| LTS vs model disagreement map + interactive risk_map.html | Sonnet | done | crash record sides with model in both cells |
| Simplified architecture figure | Claude | done | docs/architecture.png / .svg, short Mermaid |
| Submit repo link + paste 2 slides into demo deck | User | open | by 16:00 |

## Open items (next steps after the hackathon)

| Item | Why |
|---|---|
| Add exposure (bike counts, Capital Bikeshare trips) | Separate where people ride from where it is dangerous |
| Recover Ward 7-8 crashes that miss the snapshot | Equity: about a fifth of their located crashes are unmatched |
| Temporal validation (train early years, test late) | Wards are not the only shift; crashes rose 2022-2025 |
| Full 3-fit ablations (no `--fast`) | E5 deltas are near the noise floor |
