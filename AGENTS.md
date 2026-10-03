# Repository Guidelines

## Project Structure & Module Organization
Library code lives in `src/`: `src/data/` loads the RideScore snapshot and fetches crashes, `src/labels.py` builds the 0 / 1 / 2+ labels, `src/features/` builds the `ddot_`, `osmf_` and `net_` feature groups, `src/split.py` makes leave-one-ward-out folds, `src/decode.py` and `src/evaluate.py` hold the ordinal decoding and metrics, `src/models/` holds the ladder (`baselines.py` M0/M1, `spf.py` M2, `gbm.py` M3, `fusion.py` F, `xattn.py` M4 stretch), and `src/experiments.py` runs cross-validation. Entry points are numbered scripts in `scripts/` (`01_build_dataset.py` to `05_xattn.py`). Settings live in `config.yaml`; load them with `src.config.load_config`. Tests live in `tests/`. Raw data and caches stay outside the repo in `~/ridescore-data/`; `output/` holds results (CSV, PNG, the hand-in Parquet).

## Build, Test, and Development Commands
Use the repo venv for every session: `python3 -m venv .venv --system-site-packages`, then `.venv/bin/pip install geopandas statsmodels matplotlib pyyaml "ridescore @ git+https://github.com/civictechdc/ridescoredc-models@develop"`. Run the pipeline in order: `.venv/bin/python scripts/01_build_dataset.py`, `02_ladder.py`, `03_ablations.py`, `04_findings.py`; `05_xattn.py` is optional. Run tests with `.venv/bin/python -m pytest -q`. Do not `pip install` anything else mid-run without saying so.

## Coding Style & Naming Conventions
4-space indentation, `snake_case` for functions, files and config keys, `PascalCase` for classes. Join key is `dc_subblockkey`, always normalised with `.str.strip().str.lower()`. Feature columns carry a source prefix: `ddot_`, `osmf_`, `net_`. Add new features to `src/features/groups.py` so ablations pick them up. Prefer config values and CLI arguments over hard-coded paths or numbers. No formatter is configured; match the surrounding file and keep imports tidy.

## Leakage Rules
- Ward, coordinates, and anything derived from crashes (counts, injury flags, nearest-intersection fields, crash dates) are never features.
- Labels are built only from the Crashes in DC layer; features come only from the snapshot.
- Thresholds, logit biases, fusion weights and any early-stopping choice are fitted on validation wards only. Test wards are touched once per fold.
- No ward may play two roles in a fold (test, validation, train). `ward_folds` asserts this; keep the assertion.
- Every reported metric is read against M0 (majority class). Report mean +- sd over folds.

## Testing Guidelines
Tests use `pytest`. Add coverage in `tests/test_*.py` that mirrors the module under change (`test_labels.py`, `test_osm_parsing.py`, `test_decode.py`, `test_split.py`). Prefer small synthetic tables over the real data. A change to labels, decoding or splits needs a regression test.

## Documentation Rule
Four docs in `docs/` are updated in place, never duplicated: `plan.md` (plans, design decisions, Open Items table), `memory.md` (non-obvious decisions and constraints), `architecture.md` (Mermaid diagram and component table; keep in sync with code), `progress.md` (task status; update rows, do not add per-session blocks). Update whichever apply before ending a session.

## Commit & Pull Request Guidelines
Short imperative commit titles, one change each (for example `Add SPF baseline`). Do not commit `.venv/`, raw data, caches or checkpoints. The hand-in Parquet and final figures in `output/` are tracked. Describe commands run for verification in the PR body.

## Claims Discipline
Write "where crashes happen" or "where recorded crashes are expected". Never "danger per ride", "safe street" or causal language: there is no exposure data and the results are associations.
