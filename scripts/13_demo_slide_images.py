"""Render the two demo slides as 1920x1080 PNGs (slides/demo_v2_slide1.png, demo_v2_slide2.png).

Slide 1: why, question, hypotheses, what we built.  Slide 2: results per hypothesis, where we are, next.
"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAVY, GOLD, INK, MUTED = "#1E3A6E", "#B8862F", "#1F2937", "#4B5563"
GREEN, AMBER, RED, ACCENT = "#2F7D32", "#B26A00", "#B42318", "#C2410C"
TILE = "#F4F1EA"


def frame(title, subtitle=None):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 192)
    ax.set_ylim(0, 108)
    ax.axis("off")
    ax.text(6, 101.5, "RideScore DC", fontsize=24, color=NAVY, weight="bold", va="center")
    ax.text(186, 101.5, "Track: Models  ·  Challenge 1: crash risk (4.3) + new model (4.2)",
            fontsize=16, color=GOLD, weight="bold", va="center", ha="right")
    ax.plot([6, 186], [97, 97], color=GOLD, lw=2.5)
    ax.text(6, 90.5, title, fontsize=32, color=NAVY, weight="bold", va="center")
    if subtitle:
        ax.text(6, 84.5, subtitle, fontsize=16, color=MUTED, va="center")
    return fig, ax


def heading(ax, x, y, text):
    ax.text(x, y, text, fontsize=20, color=NAVY, weight="bold", va="top")
    return y - 5


def para(ax, x, y, text, width, size=16, color=INK, gap=3.3):
    for line in textwrap.wrap(text, width):
        ax.text(x, y, line, fontsize=size, color=color, va="top")
        y -= gap
    return y - 1.0


def tile(ax, x, y, w, h, big, small):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=TILE, ec="none"))
    ax.text(x + w / 2, y + h * 0.64, big, fontsize=24, color=NAVY, weight="bold",
            ha="center", va="center")
    ax.text(x + w / 2, y + h * 0.27, small, fontsize=14, color=MUTED, ha="center", va="center")


def image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")


def slide1():
    fig, ax = frame("Does street design explain where DC cyclists crash?",
                    "Yuvraj Gupta  ·  github.com/YuvrajGupta1808/CivicTech")
    y = heading(ax, 6, 79, "Why it matters")
    y = para(ax, 6, y, "RideScore colours DC streets by design (LTS, BNA). Nobody had checked "
                       "those colours against where cyclists actually crash. If they don't match, "
                       "the map shows comfort, not crash risk.", 50)
    y = heading(ax, 6, y - 1.5, "Hypotheses")
    hyps = [("H1", "Design predicts crash locations, but only modestly"),
            ("H2", "Junctions, lanes and traffic matter more than bike-lane type"),
            ("H3", "DDOT data is more useful than OSM"),
            ("H4", "Ordinal decoding (from AVB-Engage) helps find rare repeat-crash blocks"),
            ("H5", "Streets that look safe but crash are mostly at intersections")]
    for hid, text in hyps:
        ax.text(6, y, hid, fontsize=16, color=GOLD, weight="bold", va="top")
        y = para(ax, 13, y, text, 44, gap=3.1) - 0.3
    ax.text(72, 79, "What we built", fontsize=20, color=NAVY, weight="bold", va="top")
    image(fig, ROOT / "docs/architecture.png", [0.375, 0.255, 0.605, 0.47])
    tiles = [("3,089", "DC bike crashes, 2021-26"), ("19,554", "DDOT street blocks"),
             ("57", "street-design features"), ("8 wards", "each tested unseen")]
    for i, (big, small) in enumerate(tiles):
        tile(ax, 72 + i * 29, 8, 26.5, 15, big, small)
    ax.text(6, 3, "Data: © OpenStreetMap contributors (ODbL); DDOT, MPD Crashes in DC (CC BY 4.0).",
            fontsize=12, color=MUTED, va="center")
    fig.savefig(ROOT / "slides/demo_v2_slide1.png", dpi=100, facecolor="white")


def slide2():
    fig, ax = frame("What we found: design finds crash streets, comfort scores don't")
    ax.text(6, 84.5, "Held-out wards only. Share of a ward's bike crashes in the 10% of blocks "
                     "each score rates worst:", fontsize=16, color=MUTED, va="center")
    bars = [("Random", 10, "#9CA3AF"), ("LTS", 13, "#6B8BB5"), ("BNA", 18, "#6B8BB5"),
            ("Our model", 44, ACCENT)]
    for i, (name, val, col) in enumerate(bars):
        yy = 76 - i * 6.2
        ax.text(6, yy, name, fontsize=17, color=INK, va="center", weight="bold")
        ax.add_patch(FancyBboxPatch((26, yy - 2.1), val * 1.45, 4.2,
                                    boxstyle="round,pad=0,rounding_size=0.6", fc=col, ec="none"))
        ax.text(26 + val * 1.45 + 1.5, yy, f"{val}%", fontsize=18, color=col, weight="bold",
                va="center")
    y = heading(ax, 6, 49, "Hypotheses: what held")
    rows = [("H1", "Supported", GREEN, "Clearly above chance; recall of 2+ crash blocks still only 32%"),
            ("H2", "Partly", AMBER, "Junction layout and lanes matter most; bike-lane type alone barely"),
            ("H3", "Supported", GREEN, "DDOT slightly ahead of OSM; cross-attention ≈ simple concat"),
            ("H4", "Supported", GREEN, "Ordinal thresholds add +4.6 macro-F1 points to the fused model"),
            ("H5", "Rejected", RED, "79% of their crashes are mid-block, not at intersections")]
    for hid, verdict, col, text in rows:
        ax.text(6, y, hid, fontsize=16, color=GOLD, weight="bold", va="top")
        ax.text(13, y, verdict, fontsize=16, color=col, weight="bold", va="top")
        y = para(ax, 34, y, text, 48, gap=3.1) - 0.6
    x = 106
    ax.add_patch(FancyBboxPatch((x - 3, 6), 83, 78, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=TILE, ec="none"))
    y = heading(ax, x, 81, "Key evidence")
    y = para(ax, x, y, "LTS's calmest streets (LTS 1) have the most crashes per km: 6.5 vs 0.85 "
                       "for LTS 2. Worst block in DC: 14th St NW at Irving St, LTS 1 with a "
                       "protected lane, 18 crashes in 5 years. Crashes follow riders.", 52)
    y = heading(ax, x, y - 1, "Where we are")
    y = para(ax, x, y, "Risk level + score for all 28,978 street segments, risk and "
                       "disagreement maps, interactive web map, code and tests on GitHub.", 52)
    y = heading(ax, x, y - 1, "Next")
    for item in ["Show a crash layer next to LTS, not instead of it",
                 "Add bike counts to separate busy from dangerous",
                 "Fix Ward 7-8 crash matching (about 20% lost)"]:
        y = para(ax, x, y, "•  " + item, 52, gap=3.1)
    y = heading(ax, x, y - 1, "Honest limit")
    para(ax, x, y, "No ridership data: this shows where crashes happen, not danger per ride.", 52)
    ax.text(6, 3, "github.com/YuvrajGupta1808/CivicTech  ·  © OpenStreetMap contributors (ODbL); "
                  "DDOT, MPD (CC BY 4.0)", fontsize=12, color=MUTED, va="center")
    fig.savefig(ROOT / "slides/demo_v2_slide2.png", dpi=100, facecolor="white")


if __name__ == "__main__":
    (ROOT / "slides").mkdir(exist_ok=True)
    slide1()
    slide2()
    print("wrote slides/demo_v2_slide1.png, slides/demo_v2_slide2.png")
