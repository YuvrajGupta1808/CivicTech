"""Build the sub-block modelling table and the join report."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.features.build import build_table, feature_columns  # noqa: E402


def main():
    cfg = load_config()
    table, report = build_table(cfg)
    out = cfg["paths"]["output_dir"]
    table.to_parquet(cfg["paths"]["table"], index=False)
    with open(out / "join_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    by_ward = table.groupby("ward")["level"].value_counts().unstack(fill_value=0)
    by_ward["crashes"] = table.groupby("ward")["crash_count"].sum()
    by_ward.to_csv(out / "labels_by_ward.csv")
    feats = feature_columns(table)
    print(f"sub-blocks: {len(table)}  features: {len(feats)}")
    print("levels:", table["level"].value_counts().sort_index().to_dict())
    print(by_ward)
    print(json.dumps({k: v for k, v in report.items() if k != "dropped_feature_cols"}, indent=1,
                     default=str))
    print("dropped:", report["dropped_feature_cols"])


if __name__ == "__main__":
    main()
