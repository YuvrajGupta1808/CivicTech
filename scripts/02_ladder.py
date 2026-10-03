"""E1 model ladder + E3 decoders + E9 fusion weights, leave-one-ward-out."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluate import summarise  # noqa: E402
from src.experiments import run_cv  # noqa: E402
from src.features.build import feature_columns  # noqa: E402

SHOW = ["accuracy", "macro_f1", "f1_2", "rec_2", "mae", "auc_any", "auc_repeat",
        "top10_capture_count", "top10_capture_length"]
ORDER = ["m0", "m1", "spf", "gbm", "xattn", "xattn_concat", "fusion", "fusion_equal",
         "fusion_valsel"]
LABELS = {"m0": "M0 majority", "m1": "M1 LTS v1", "spf": "M2 SPF", "gbm": "M3 GBM",
          "xattn": "M4 cross-attn", "xattn_concat": "M4 concat", "fusion": "F fusion",
          "fusion_equal": "F equal wts", "fusion_valsel": "F val wts"}


def load_external(out):
    path = out / "xattn_probs.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    ext = {name: df[df["variant"] == variant].drop(columns="variant")
           for name, variant in [("xattn", "xattn"), ("xattn_concat", "concat")]}
    print(f"[ladder] using M4 cross-attention predictions ({len(ext['xattn'])} rows)")
    return ext


def ladder_plot(summary, out):
    best = (summary.sort_values("macro_f1_mean", ascending=False)
            .drop_duplicates("model").set_index("model"))
    best = best.loc[[m for m in ORDER if m in best.index]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, metric, label in [(axes[0], "macro_f1", "Macro-F1 (best decoder)"),
                              (axes[1], "top10_capture_count", "Crashes in top-10% sub-blocks")]:
        ax.barh([LABELS[m] for m in best.index[::-1]], best[f"{metric}_mean"][::-1],
                xerr=best[f"{metric}_std"][::-1], color="#4c78a8")
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axvline(best.loc["m0", "macro_f1_mean"], color="grey", ls="--", lw=1)
    axes[1].axvline(0.10, color="grey", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=160)


def main():
    cfg = load_config()
    out = cfg["paths"]["output_dir"]
    table = pd.read_parquet(cfg["paths"]["table"])
    feats = feature_columns(table)
    print(f"[ladder] {len(table)} sub-blocks, {len(feats)} features")
    metrics, oof = run_cv(table, feats, label_col="level", cfg=cfg, external=load_external(out))
    metrics.to_csv(out / "metrics_folds.csv", index=False)
    oof.to_parquet(out / "oof_predictions.parquet", index=False)
    summary = summarise(metrics)
    summary.to_csv(out / "metrics_ladder.csv", index=False)
    cols = ["model", "decoder"] + [f"{m}_mean" for m in SHOW if f"{m}_mean" in summary]
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(summary[cols].to_string(index=False))
    main_model = "fusion" if "fusion" in set(metrics["model"]) else "gbm"
    per_ward = metrics[(metrics["model"] == main_model) & (metrics["decoder"] == "thresholds")]
    per_ward.to_csv(out / "metrics_by_ward.csv", index=False)
    ladder_plot(summary, out)


if __name__ == "__main__":
    main()
