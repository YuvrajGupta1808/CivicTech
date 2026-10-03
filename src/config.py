"""Load config.yaml and expand user paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    for key, value in paths.items():
        p = Path(value).expanduser()
        paths[key] = p if p.is_absolute() else ROOT / p
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    return cfg
