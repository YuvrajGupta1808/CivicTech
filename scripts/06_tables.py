"""Render result CSVs in output/ as markdown tables (output/tables.md) for the README."""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "output"
NAMES = {"m0": "M0 majority", "m1": "M1 RideScore LTS v1", "spf": "M2 SPF (neg. binomial)",
         "gbm": "M3 gradient boosting", "xattn": "M4 OSM-DDOT cross-attention",
         "fusion": "F fusion (fixed weights)", "fusion_equal": "F fusion (equal weights)",
         "fusion_valsel": "F fusion (val-selected weights)"}
ORDER = list(NAMES)


def pm(df, m):
    return df[f"{m}_mean"].map("{:.3f}".format) + " ± " + df[f"{m}_std"].map("{:.3f}".format)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def ladder_table(s):
    best = s[(s.decoder == "thresholds") | s.model.isin(["m0", "m1"])]
    best = best[best.model.isin(ORDER)].copy()
    best["order"] = best.model.map(ORDER.index)
    best = best.sort_values("order")
    return pd.DataFrame({
        "Model": best.model.map(NAMES), "Decoder": best.decoder,
        "Macro-F1": pm(best, "macro_f1"), "Recall lvl 2": pm(best, "rec_2"),
        "AUC any crash": pm(best, "auc_any"), "AUC 2+ crashes": pm(best, "auc_repeat"),
        "Top-10% capture": pm(best, "top10_capture_count"),
        "Top-10% capture (by length)": pm(best, "top10_capture_length"),
        "Accuracy": best["accuracy_mean"].map("{:.3f}".format)})


def decoder_table(s):
    d = s[s.model.isin(["spf", "gbm", "xattn", "fusion"])]
    piv = d.pivot(index="model", columns="decoder", values="macro_f1_mean")
    piv = piv.loc[[m for m in ORDER if m in piv.index]]
    return pd.DataFrame({"Model": [NAMES[m] for m in piv.index],
                         "argmax": piv["argmax"].map("{:.3f}".format).values,
                         "logit bias (val)": piv["bias"].map("{:.3f}".format).values,
                         "expected-level thresholds (val)":
                             piv["thresholds"].map("{:.3f}".format).values})


def per_ward_table(folds, model):
    f = folds[(folds.model == model) & (folds.decoder == "thresholds")].sort_values("test_ward")
    return pd.DataFrame({"Test ward": f.test_ward, "Sub-blocks": f.n, "Level-2": f.n_pos2,
                         "Macro-F1": f.macro_f1.map("{:.3f}".format),
                         "AUC any": f.auc_any.map("{:.3f}".format),
                         "Top-10% capture": f.top10_capture_count.map("{:.3f}".format)})


def main():
    s = pd.read_csv(OUT / "metrics_ladder.csv")
    folds = pd.read_csv(OUT / "metrics_folds.csv")
    main_model = "fusion" if "fusion" in set(s.model) else "gbm"
    parts = ["## E1 ladder\n", md(ladder_table(s)), "\n## E3 decoders (macro-F1)\n",
             md(decoder_table(s)), f"\n## Per ward ({main_model}, thresholds)\n",
             md(per_ward_table(folds, main_model))]
    for name in ["ablation_groups", "ablation_sources", "label_variants", "robustness",
                 "lts_vs_crashes", "surprise_breakdown", "xattn_summary"]:
        p = OUT / f"{name}.csv"
        if p.exists():
            parts += [f"\n## {name}\n", md(pd.read_csv(p).round(3))]
    (OUT / "tables.md").write_text("\n".join(parts) + "\n")
    print((OUT / "tables.md").read_text()[:6000])


if __name__ == "__main__":
    sys.exit(main())
