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


def bullets(ax, x, y, items, width, size=17, gap=3.4):
    for item in items:
        lines = textwrap.wrap(item, width)
        ax.text(x, y, "•", fontsize=size, color=GOLD, weight="bold", va="top")
        for line in lines:
            ax.text(x + 3, y, line, fontsize=size, color=INK, va="top")
            y -= gap
        y -= 1.4
    return y


def slide1():
    fig, ax = frame("Does street design explain where DC cyclists crash?",
                    "Yuvraj Gupta  ·  github.com/YuvrajGupta1808/CivicTech")
    y = heading(ax, 6, 76, "Why it matters")
    y = para(ax, 6, y, "RideScore colours DC streets by design (LTS, BNA). Nobody had checked "
                       "those colours against where cyclists actually crash.", 44, size=17, gap=3.5)
    y = heading(ax, 6, y - 2, "What we did")
    bullets(ax, 6, y, ["Joined 3,089 DC bike crashes (2021-26) to 19,554 DDOT street blocks",
                       "57 street-design facts from OpenStreetMap + DDOT",
                       "3 models, incl. OSM-DDOT cross-attention from AVB-Engage, fused",
                       "Tested only on wards the model never saw"], 40)
    image(fig, ROOT / "docs/architecture.png", [0.33, 0.12, 0.655, 0.62])
    ax.text(6, 3, "Data: © OpenStreetMap contributors (ODbL); DDOT, MPD Crashes in DC (CC BY 4.0).",
            fontsize=12, color=MUTED, va="center")
    fig.savefig(ROOT / "slides/demo_v3_slide1.png", dpi=100, facecolor="white")


def slide2():
    fig, ax = frame("What we found: comfort scores don't show where cyclists crash")
    image(fig, ROOT / "output/lts_disagreement_map.png", [0.02, 0.04, 0.40, 0.75])
    x = 86
    ax.text(x, 80, "Share of crashes in the 10% of blocks each score rates worst",
            fontsize=17, color=NAVY, weight="bold", va="top")
    ax.text(x, 75.5, "(wards the model never saw)", fontsize=14, color=MUTED, va="top")
    bars = [("Random", 10, "#9CA3AF"), ("LTS", 13, "#6B8BB5"), ("BNA", 18, "#6B8BB5"),
            ("Our model", 44, ACCENT)]
    for i, (name, val, col) in enumerate(bars):
        yy = 67.5 - i * 5.6
        ax.text(x, yy, name, fontsize=16, color=INK, va="center", weight="bold")
        ax.add_patch(FancyBboxPatch((x + 20, yy - 1.9), val * 1.55, 3.8,
                                    boxstyle="round,pad=0,rounding_size=0.6", fc=col, ec="none"))
        ax.text(x + 20 + val * 1.55 + 1.5, yy, f"{val}%", fontsize=17, color=col, weight="bold",
                va="center")
    y = heading(ax, x, 44, "Key evidence")
    y = bullets(ax, x, y, ["Orange streets: LTS calls them calm, our model flags them. 4% of the "
                           "network, 15% of all bike crashes",
                           "Worst block: 14th St NW at Irving St. LTS 1, protected lane, 18 crashes"],
                52, size=16, gap=3.2)
    y = heading(ax, x, y - 0.5, "Next")
    y = bullets(ax, x, y, ["Crash layer next to LTS, not instead of it",
                           "Bike counts to separate busy from dangerous"], 52, size=16, gap=3.2)
    ax.text(x, y - 0.5, "Limit: no ridership data, so this shows where crashes happen, "
            "not danger per ride.", fontsize=14, color=MUTED, va="top", style="italic")
    ax.text(6, 3, "github.com/YuvrajGupta1808/CivicTech  ·  © OpenStreetMap contributors (ODbL); "
                  "DDOT, MPD (CC BY 4.0)", fontsize=12, color=MUTED, va="center")
    fig.savefig(ROOT / "slides/demo_v3_slide2.png", dpi=100, facecolor="white")


if __name__ == "__main__":
    (ROOT / "slides").mkdir(exist_ok=True)
    slide1()
    slide2()
    print("wrote slides/demo_v3_slide1.png, slides/demo_v3_slide2.png")
