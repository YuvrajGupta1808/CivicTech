"""Render the two demo slides as 1920x1080 PNGs (slides/demo_slide1.png, demo_slide2.png)."""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAVY, GOLD, INK, MUTED, ACCENT = "#1E3A6E", "#B8862F", "#1F2937", "#4B5563", "#C2410C"


def frame(title):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 192)
    ax.set_ylim(0, 108)
    ax.axis("off")
    ax.text(6, 101, "RideScore DC", fontsize=26, color=NAVY, weight="bold", va="center")
    ax.text(186, 101, "Track: Models  ·  Challenge 1 (ideas 4.3 + 4.2)", fontsize=17, color=GOLD,
            weight="bold", va="center", ha="right")
    ax.plot([6, 186], [96.5, 96.5], color=GOLD, lw=2.5)
    ax.text(6, 90, title, fontsize=34, color=NAVY, weight="bold", va="center")
    return fig, ax


def block(ax, x, y, width_chars, head, body, size=17, gap=3.4):
    ax.text(x, y, head, fontsize=size + 3, color=NAVY, weight="bold", va="top")
    y -= 4.4
    for para in body:
        for line in textwrap.wrap(para, width_chars):
            ax.text(x, y, line, fontsize=size, color=INK, va="top")
            y -= gap
        y -= 1.2
    return y - 1.5


def image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def footer(ax, text):
    ax.text(6, 3, text, fontsize=13, color=MUTED, va="center")


def slide1():
    fig, ax = frame("Calm on the map, crashes on the street")
    ax.text(6, 83.5, "Yuvraj Gupta  ·  github.com/YuvrajGupta1808/CivicTech", fontsize=16, color=MUTED,
            va="center")
    y = block(ax, 6, 78, 58, "The problem",
              ["RideScore colours DC streets by design (LTS). Nobody had checked those colours "
               "against where cyclists actually crash."])
    y = block(ax, 6, y, 58, "What we did",
              ["Joined 3,089 DC bike crashes (Crashes in DC, 2021-2026) to 19,554 DDOT blocks.",
               "Trained a crash model on OSM + DDOT street design: OSM-DDOT cross-attention "
               "+ boosting + traffic-safety formula. Tested only on wards it never saw."])
    ax.text(6, y, "What we found", fontsize=20, color=NAVY, weight="bold", va="top")
    ax.text(6, y - 6, "44%", fontsize=58, color=ACCENT, weight="bold", va="top")
    ax.text(31, y - 6.5, "of bike crashes sit in the 10% of blocks", fontsize=17, color=INK, va="top")
    ax.text(31, y - 10, "our model ranks riskiest.", fontsize=17, color=INK, va="top")
    ax.text(31, y - 14.5, "LTS: 13%   ·   BNA: 18%   ·   random: 10%", fontsize=17, color=MUTED,
            va="top", weight="bold")
    image(fig, ROOT / "output/bna_compare.png", [0.50, 0.17, 0.47, 0.62])
    ax.text(144, 15, "Crashes per km on 'comfortable' vs 'uncomfortable' streets", fontsize=14,
            color=MUTED, ha="center")
    footer(ax, "Data: © OpenStreetMap contributors (ODbL); DDOT, MPD Crashes in DC (CC BY 4.0).  "
               "Associations, not causes.")
    fig.savefig(ROOT / "slides/demo_slide1.png", dpi=100, facecolor="white")


def slide2():
    fig, ax = frame("Where LTS says calm, crashes say otherwise")
    image(fig, ROOT / "output/lts_disagreement_map.png", [0.03, 0.05, 0.42, 0.70])
    x = 96
    y = block(ax, x, 82, 54, "Evidence",
              ["Streets LTS calls calm but our model flags: 4% of the network, 15% of all bike "
               "crashes (5.3 crashes/km, same as streets both call risky).",
               "Worst block in DC: 14th St NW at Irving St. LTS 1 with a protected lane, "
               "18 crashes in 5 years."])
    y = block(ax, x, y, 54, "Honest limits",
              ["No ridership data: this shows where crashes happen, not danger per ride. Busy "
               "bike routes get more crashes.",
               "Wards 7-8 lose about 20% of crashes in the data match."])
    block(ax, x, y, 54, "Next steps for RideScore",
          ["Add a crash layer next to LTS, not instead of it.",
           "Add bike counts to separate busy from dangerous.",
           "Fix Ward 7-8 crash matching. Per-segment scores ready: crash_risk_by_segment.parquet."])
    footer(ax, "github.com/YuvrajGupta1808/CivicTech  ·  © OpenStreetMap contributors (ODbL); "
               "DDOT, MPD (CC BY 4.0)")
    fig.savefig(ROOT / "slides/demo_slide2.png", dpi=100, facecolor="white")


if __name__ == "__main__":
    (ROOT / "slides").mkdir(exist_ok=True)
    slide1()
    slide2()
    print("wrote slides/demo_slide1.png, slides/demo_slide2.png")
