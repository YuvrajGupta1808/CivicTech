"""Draw the one-page architecture figure (docs/architecture.png and .svg)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs"
RAMPS = {  # fill, edge, text (light ramps; the figure has its own white background)
    "teal": ("#E1F5EE", "#0F6E56", "#085041"),
    "coral": ("#FAECE7", "#993C1D", "#712B13"),
    "purple": ("#EEEDFE", "#534AB7", "#3C3489"),
    "gray": ("#F1EFE8", "#5F5E5A", "#2C2C2A"),
}


def box(ax, x, y, w, h, title, sub, ramp, title_size=11.5):
    fill, edge, text = RAMPS[ramp]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=1.0))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=title_size,
                color=text, weight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=9.5,
                color=edge, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                color=text, weight="bold", linespacing=1.4)
    return (x, y, w, h)


def stack(ax, b, parts, ramp):
    """Write (title, subtitle) pairs top to bottom inside a tall box."""
    fill, edge, text = RAMPS[ramp]
    x, y, w, h = b
    slot = h / len(parts)
    for i, (title, sub) in enumerate(parts):
        cy = y + h - slot * (i + 0.5)
        ax.text(x + w / 2, cy + 1.6, title, ha="center", va="center", fontsize=11, color=text,
                weight="bold")
        ax.text(x + w / 2, cy - 1.6, sub, ha="center", va="center", fontsize=9.5, color=edge,
                linespacing=1.35)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                 color="#888780", lw=1.0, shrinkA=0, shrinkB=0))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b, dy=0.0):
    return (b[0], b[1] + b[3] / 2 + dy)


def main():
    fig, ax = plt.subplots(figsize=(14, 5.7))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42.5)
    ax.axis("off")

    heads = [(9, "Public data"), (30, "One table"), (52, "Three learned models"),
             (73, "Fuse + decode"), (91.5, "Outputs")]
    for x, label in heads:
        ax.text(x, 40.6, label, ha="center", va="center", fontsize=12.5, color="#2C2C2A",
                weight="bold")

    h = 8
    osm = box(ax, 1, 30, 16, h, "OpenStreetMap", "bike lanes, oneway,\nmapped speed", "teal")
    ddot = box(ax, 1, 19.5, 16, h, "DDOT SubBlock", "lanes, speed limit,\ntraffic, pavement",
               "teal")
    crash = box(ax, 1, 9, 16, h, "Crashes in DC", "3,089 bike crashes\n2021-2026", "coral")

    table = box(ax, 22, 9, 16, 29, "", None, "gray")
    stack(ax, table, [("19,554 sub-blocks", "one row per\nDDOT block segment"),
                      ("57 design features", "DDOT + OSM +\nnetwork geometry"),
                      ("Label 0 / 1 / 2+", "bike crashes\nin 5 years")], "gray")

    spf = box(ax, 43, 30, 18, h, "M2  SPF", "negative binomial\n(traffic-safety standard)",
              "purple")
    gbm = box(ax, 43, 19.5, 18, h, "M3  Gradient boosting", "57 features,\nclass-balanced",
              "purple")
    xat = box(ax, 43, 9, 18, h, "M4  Cross-attention", "OSM tokens \u2194 DDOT tokens\n(AVT-CA / AVB-Engage)",
              "purple")

    fuse = box(ax, 66, 13.5, 14, 20, "", None, "purple")
    stack(ax, fuse, [("Late fusion", "0.4 M4 + 0.4 M3\n+ 0.2 M2"),
                     ("Ordinal decoding", "thresholds fit on\nvalidation wards")], "purple")

    out1 = box(ax, 84.5, 30, 14.5, h, "Risk per segment", "level 0/1/2 + score\n28,978 segments",
               "gray", title_size=11)
    out2 = box(ax, 84.5, 19.5, 14.5, h, "Maps", "risk, surprises,\nLTS disagreement", "gray",
               title_size=11)
    out3 = box(ax, 84.5, 9, 14.5, h, "Checks", "vs majority,\nLTS and BNA", "gray",
               title_size=11)

    for b in (osm, ddot, crash):
        arrow(ax, right(b), (table[0], b[1] + h / 2))
    for b in (spf, gbm, xat):
        arrow(ax, (table[0] + table[2], b[1] + h / 2), left(b))
        arrow(ax, right(b), (fuse[0], b[1] + h / 2 if fuse[1] < b[1] + h / 2 < fuse[1] + fuse[3]
                             else fuse[1] + fuse[3] / 2))
    for b in (out1, out2, out3):
        arrow(ax, (fuse[0] + fuse[2], b[1] + h / 2 if fuse[1] < b[1] + h / 2 < fuse[1] + fuse[3]
                   else fuse[1] + fuse[3] / 2), left(b))

    ax.add_patch(FancyBboxPatch((22, 1.2), 77, 5, boxstyle="round,pad=0,rounding_size=1.0",
                                fc="white", ec="#888780", lw=0.9, ls=(0, (4, 3))))
    ax.text(60.5, 3.7, "Leave-one-ward-out:  train on 5 wards,  fit thresholds on 2,  test on 1.  "
            "Every score comes from a model that never saw that ward.",
            ha="center", va="center", fontsize=10.5, color="#444441")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "architecture.png", dpi=170, facecolor="white")
    fig.savefig(OUT / "architecture.svg", facecolor="white")
    print("wrote", OUT / "architecture.png", OUT / "architecture.svg")


if __name__ == "__main__":
    main()
