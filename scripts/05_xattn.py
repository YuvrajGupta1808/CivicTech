#!/usr/bin/env python
"""M4 (stretch): OSM <-> DDOT cross-attention Transformer, leave-one-ward-out CV (same folds as the GBM ladder).

For every fold (test ward, 2 cyclic validation wards, 5 train wards) and mode in (xattn, concat):
train N seeds, keep each seed's best epoch (val macro-F1 after threshold decoding), predict val + test
(seed-averaged). For `xattn` also predict with the OSM / DDOT stream ablated at inference (E7).
Writes output/xattn_probs.parquet and output/xattn_summary.csv.

  python scripts/05_xattn.py                       # real table (~/ridescore-data/subblock_table.parquet)
  python scripts/05_xattn.py --synthetic --epochs 3 --seeds 1   # smoke test on a synthetic table
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TABLE = Path.home() / "ridescore-data" / "subblock_table.parquet"
# config.yaml xattn section (hardcoded: the GPU env has no pyyaml)
CFG = dict(d_model=32, heads=4, exchanges=2, dropout=0.1, source_dropout=0.15, ordinal_lambda=0.15,
           label_smoothing=0.1, lr=1e-3, weight_decay=0.05, batch_size=256)
VARIANTS = ("xattn", "concat", "xattn_no_osm", "xattn_no_ddot")


def make_synthetic(path, n=19554, seed=0):
    """Synthetic table with the subblock_table schema (ddot_/osmf_/net_ features, ward, level)."""
    rng = np.random.default_rng(seed)
    ward = rng.integers(1, 9, n)
    latent = rng.normal(size=n)
    df = pd.DataFrame({"dc_subblockkey": [f"{i:032x}" for i in range(n)], "ward": ward})
    for j in range(24):  # DDOT numeric
        x = latent * rng.normal(0, 0.4) + rng.normal(size=n)
        x[rng.random(n) < rng.uniform(0, 0.5)] = np.nan
        df[f"ddot_num{j}"] = x
    for j in range(4):  # DDOT categorical
        df[f"ddot_cat{j}"] = pd.Categorical(rng.choice(list("abcde")[: 3 + j % 3], n, p=None))
    for j in range(18):  # OSM numeric
        x = latent * rng.normal(0, 0.3) + rng.normal(size=n)
        x[rng.random(n) < rng.uniform(0, 0.4)] = np.nan
        df[f"osmf_num{j}"] = x
    for j in range(3):  # OSM categorical (object strings)
        df[f"osmf_cat{j}"] = rng.choice(["none", "lane", "track", "shared"], n)
        df.loc[rng.random(n) < 0.1, f"osmf_cat{j}"] = None
    for j in range(6):
        df[f"net_num{j}"] = latent * 0.3 + rng.normal(size=n)
    score = 0.9 * latent + 0.5 * df["net_num0"] - 3.2 + rng.normal(0, 0.6, n)
    df["level"] = np.digitize(score, [np.quantile(score, 0.912), np.quantile(score, 0.974)])
    df.to_parquet(path)
    return path


def run_fold(task):
    """Worker: all modes x seeds for one fold. Returns a DataFrame of probabilities."""
    table_path, fold_i, modes, seeds, epochs, device = task
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    from src.models.xattn import TabularEncoder, fit_model, predict_proba
    from src.split import ward_folds

    df = pd.read_parquet(table_path)
    fold = ward_folds(df["ward"].to_numpy())[fold_i]
    tr, va, te = df.iloc[fold.train_idx], df.iloc[fold.val_idx], df.iloc[fold.test_idx]
    enc = TabularEncoder(df).fit(tr)
    dev = torch.device(device)
    D = {k: enc.transform(x, dev) for k, x in (("train", tr), ("val", va), ("test", te))}
    y_tr, y_va = tr["level"].to_numpy().astype(int), va["level"].to_numpy().astype(int)
    keys = {"val": va["dc_subblockkey"].to_numpy(), "test": te["dc_subblockkey"].to_numpy()}
    t0 = time.time()
    acc = {}  # (variant, split) -> list of probs over seeds
    for mode in modes:
        ablations = ("none", "no_osm", "no_ddot") if mode == "xattn" else ("none",)
        for seed in seeds:
            model, info = fit_model(enc, D["train"], y_tr, D["val"], y_va, mode, CFG, seed, dev, epochs)
            print(f"[test_ward {fold.test_ward}] {mode} seed {seed} best_ep {info['best_epoch']} "
                  f"val_f1 {info['val_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            for ab in ablations:
                variant = mode if ab == "none" else f"{mode}_{ab}"
                for sp in ("val", "test"):
                    acc.setdefault((variant, sp), []).append(predict_proba(model, D[sp], ab))
            del model
            torch.cuda.empty_cache()
    rows = []
    for (variant, sp), ps in acc.items():
        P = np.mean(ps, axis=0)
        out = pd.DataFrame({"dc_subblockkey": keys[sp], "test_ward": fold.test_ward, "split": sp,
                            "variant": variant, "p0": P[:, 0], "p1": P[:, 1], "p2": P[:, 2]})
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarise(probs: pd.DataFrame, labels: pd.Series):
    from sklearn.metrics import roc_auc_score
    from src.models.xattn import decode, expected_level, fit_thresholds_macro_f1, macro_f1

    y_of = labels
    per_fold = []
    pooled = {v: ([], []) for v in VARIANTS}
    for (variant, tw), g in probs.groupby(["variant", "test_ward"]):
        val, test = g[g.split == "val"], g[g.split == "test"]
        mv = expected_level(val[["p0", "p1", "p2"]].to_numpy())
        mt = expected_level(test[["p0", "p1", "p2"]].to_numpy())
        yv, yt = y_of.loc[val.dc_subblockkey].to_numpy(), y_of.loc[test.dc_subblockkey].to_numpy()
        thr, vf1 = fit_thresholds_macro_f1(mv, yv)
        yhat = decode(mt, thr)
        auc = roc_auc_score(yt > 0, mt) if 0 < (yt > 0).sum() < len(yt) else np.nan
        pooled[variant][0].append(yt)
        pooled[variant][1].append(yhat)
        per_fold.append(dict(variant=variant, test_ward=tw, val_f1=vf1, test_f1=macro_f1(yt, yhat), auc_any=auc,
                             test_acc=float((yhat == yt).mean()),
                             maj_f1=macro_f1(yt, np.zeros_like(yt)), thr1=thr[0], thr2=thr[1]))
    pf = pd.DataFrame(per_fold)
    # E7: ablated test predictions decoded with the *full model's* validation thresholds (no re-fit)
    fixed = {}
    for (variant, tw), g in probs[probs.variant.isin(["xattn", "xattn_no_osm", "xattn_no_ddot"])].groupby(["variant", "test_ward"]):
        full = pf[(pf.variant == "xattn") & (pf.test_ward == tw)].iloc[0]
        test = g[g.split == "test"]
        yt = y_of.loc[test.dc_subblockkey].to_numpy()
        fixed[(variant, tw)] = macro_f1(yt, decode(expected_level(test[["p0", "p1", "p2"]].to_numpy()),
                                                   [full.thr1, full.thr2]))
    pf["test_f1_fullthr"] = [fixed.get((v, w), np.nan) for v, w in zip(pf.variant, pf.test_ward)]
    summ = []
    for v in VARIANTS:
        s = pf[pf.variant == v]
        if s.empty:
            continue
        yy, hh = np.concatenate(pooled[v][0]), np.concatenate(pooled[v][1])
        summ.append(dict(variant=v, n_folds=len(s), test_macro_f1_mean=s.test_f1.mean(), test_macro_f1_sd=s.test_f1.std(),
                         auc_any_mean=s.auc_any.mean(), auc_any_sd=s.auc_any.std(), test_acc_mean=s.test_acc.mean(),
                         macro_f1_pooled_oof=macro_f1(yy, hh), val_macro_f1_mean=s.val_f1.mean(),
                         test_macro_f1_fullthr_mean=s.test_f1_fullthr.mean(), majority_macro_f1_mean=s.maj_f1.mean()))
    return pd.DataFrame(summ), pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    ap.add_argument("--synthetic", action="store_true", help="generate + use a synthetic table")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--modes", nargs="+", default=["xattn", "concat"])
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="fold indices (default all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--tag", default="", help="suffix for output names (e.g. _synth)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = a.table
    if a.synthetic:
        scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "xattn_synth_table.parquet"
        table = str(make_synthetic(scratch))
        a.tag = a.tag or "_synth"
    df = pd.read_parquet(table, columns=["dc_subblockkey", "ward", "level"])
    from src.split import ward_folds
    n_folds = len(ward_folds(df["ward"].to_numpy()))
    folds = a.folds if a.folds is not None else list(range(n_folds))
    labels = df.set_index("dc_subblockkey")["level"].astype(int)
    print(f"table {table}: {len(df)} rows, labels {np.bincount(labels)}, folds {folds}, modes {a.modes}, "
          f"seeds {a.seeds}, epochs {a.epochs}, workers {a.workers}", flush=True)

    t0 = time.time()
    tasks = [(table, f, tuple(a.modes), tuple(a.seeds), a.epochs, a.device) for f in folds]
    with cf.ProcessPoolExecutor(max_workers=min(a.workers, len(tasks)), mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(run_fold, tasks))
    probs = pd.concat(results, ignore_index=True)
    probs.to_parquet(out / f"xattn_probs{a.tag}.parquet", index=False)
    summ, pf = summarise(probs, labels)
    summ.to_csv(out / f"xattn_summary{a.tag}.csv", index=False)
    pf.to_csv(out / f"xattn_per_fold{a.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
