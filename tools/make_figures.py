"""
Draw the diagram figures for the proposal, in the document's colour scheme.

    python make_figures.py

Writes figures/pipeline.png, figures/crossencoder.png, figures/ann.png, figures/cnn.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")
OUT = "figures"
NAVY = "#1F3864"
BLUE = "#2E74B5"
LIGHT = "#EAF1F8"
MID = "#BDD3EA"
GREY = "#595959"
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, tc=NAVY, size=9, bold=False):
    """A box occupying exactly (x, y, w, h) - no padding, so gaps stay as laid out."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=tc,
            zorder=3, fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, p1, p2, color=BLUE, lw=1.3):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, color=color,
                                 linewidth=lw, shrinkA=0, shrinkB=0, zorder=1))


def elbow(ax, x1, y1, x2, y2, color=BLUE, lw=1.3):
    """Right-angle connector: across, then down/up into the target."""
    ax.plot([x1, x2], [y1, y1], color=color, lw=lw, zorder=1, solid_capstyle="round")
    arrow(ax, (x2, y1), (x2, y2), color=color, lw=lw)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {OUT}/{name}")


def title(ax, text, sub=None):
    ax.text(0.5, 1.0, text, ha="center", va="top", fontsize=11, color=NAVY, fontweight="bold")
    if sub:
        ax.text(0.5, 0.0, sub, ha="center", va="bottom", fontsize=8.5, color=GREY, style="italic")


# ------------------------------------------------------------------ pipeline
def pipeline():
    """Vertical flow - unambiguous, and it fits a portrait page without crowding."""
    fig, ax = canvas(7.4, 6.6)
    H, STEP = 0.085, 0.135
    ys = [0.855 - i * STEP for i in range(7)]

    box(ax, 0.07, ys[0], 0.37, H, "Syllabus\nPDF · DOCX", fc="white", bold=True, size=9)
    box(ax, 0.56, ys[0], 0.37, H, "Course material\nPDF · PPTX · DOCX", fc="white", bold=True,
        size=9)
    box(ax, 0.07, ys[1], 0.37, H, "Unit and topic extraction", size=8.5)
    box(ax, 0.56, ys[1], 0.37, H, "Page, slide and section\ntext extraction", size=8.5)
    box(ax, 0.14, ys[2], 0.72, H, "Pair generation — every topic against every page", size=9)
    box(ax, 0.14, ys[3], 0.72, H, "Automatic labelling  →  split by subject area", size=9, fc=MID)
    box(ax, 0.14, ys[4], 0.72, H, "Model training — four neural networks",
        fc=NAVY, ec=NAVY, tc="white", bold=True, size=9.5)
    box(ax, 0.14, ys[5], 0.72, H,
        "Evaluation on an unseen subject area\nand against human-verified labels", fc=MID, size=8.5)
    box(ax, 0.02, ys[6], 0.96, H,
        "Per-unit coverage report\nproperly taught · barely taught · not taught, with evidence",
        ec=NAVY, bold=True, size=8.5)

    arrow(ax, (0.255, ys[0]), (0.255, ys[1] + H))
    arrow(ax, (0.745, ys[0]), (0.745, ys[1] + H))
    mid = ys[1] - (STEP - H) / 2
    ax.plot([0.255, 0.745], [mid] * 2, color=BLUE, lw=1.3, zorder=1)
    ax.plot([0.255, 0.255], [ys[1], mid], color=BLUE, lw=1.3, zorder=1)
    ax.plot([0.745, 0.745], [ys[1], mid], color=BLUE, lw=1.3, zorder=1)
    arrow(ax, (0.5, mid), (0.5, ys[2] + H))
    for i in (2, 3, 4, 5):
        arrow(ax, (0.5, ys[i]), (0.5, ys[i + 1] + H))

    title(ax, "SyllabiSense — end-to-end pipeline")
    save(fig, "pipeline.png")


# ------------------------------------------------------------ cross-encoder
def crossencoder():
    fig, ax = canvas(9.4, 3.3)
    box(ax, 0.02, 0.60, 0.15, 0.17, "Topic text", fc="white", bold=True, size=9)
    box(ax, 0.02, 0.30, 0.15, 0.17, "Page text", fc="white", bold=True, size=9)
    box(ax, 0.235, 0.465, 0.135, 0.17, "[CLS] topic\n[SEP] page", size=8.5)
    box(ax, 0.425, 0.465, 0.145, 0.17, "Token + position\nembeddings", size=8.5)
    for i in range(3):
        box(ax, 0.625 + i * 0.058, 0.42, 0.044, 0.26, "", fc=MID)
    ax.text(0.712, 0.74, "× 6", ha="center", va="bottom", fontsize=10, color=NAVY,
            fontweight="bold")
    ax.text(0.712, 0.37, "transformer blocks\nself-attention + feed-forward\nall layers trained",
            ha="center", va="top", fontsize=8, color=GREY, linespacing=1.4)
    box(ax, 0.845, 0.465, 0.135, 0.17, "Linear\n→ sigmoid", fc=NAVY, ec=NAVY, tc="white",
        bold=True, size=9)
    ax.text(0.9125, 0.42, "P(page is evidence)", ha="center", va="top", fontsize=8, color=GREY)

    elbow(ax, 0.17, 0.685, 0.205, 0.59)
    elbow(ax, 0.17, 0.385, 0.205, 0.51)
    ax.plot([0.205, 0.235], [0.55, 0.55], color=BLUE, lw=1.3, zorder=1)
    arrow(ax, (0.205, 0.55), (0.235, 0.55))
    arrow(ax, (0.37, 0.55), (0.425, 0.55))
    arrow(ax, (0.57, 0.55), (0.625, 0.55))
    arrow(ax, (0.785, 0.55), (0.845, 0.55))

    title(ax, "Model 1 — Transformer cross-encoder (fine-tuned end to end)",
          "Self-attention relates every word of the topic to every word of the page, so the "
          "decision is not read off a fixed similarity score.")
    save(fig, "crossencoder.png")


# ---------------------------------------------------------------------- ANN
def ann():
    fig, ax = canvas(8.6, 3.9)
    stack = [("Input — 387 features", LIGHT,
              "TF-IDF reduced to 128 by SVD · cosine similarity · word overlap · length"),
             ("Dense 256  +  ReLU", MID, "learns combinations of the topic and page features"),
             ("Dropout 0.3", "white", "regularisation during training only"),
             ("Dense 64  +  ReLU", MID, "compresses to a smaller decision space"),
             ("Dropout 0.2", "white", "regularisation during training only"),
             ("Dense 1  +  Sigmoid", NAVY, "probability that the page is evidence for the topic")]
    H, STEP = 0.088, 0.137
    for i, (label, fc, note) in enumerate(stack):
        y = 0.80 - i * STEP
        box(ax, 0.04, y, 0.33, H, label, fc=fc, tc="white" if fc == NAVY else NAVY, bold=True,
            size=9)
        ax.text(0.40, y + H / 2, note, ha="left", va="center", fontsize=8.5, color=GREY)
        if i < len(stack) - 1:
            arrow(ax, (0.205, y), (0.205, y - (STEP - H)))
    title(ax, "Model 2 — Evidence ANN (feature-based comparison)")
    save(fig, "ann.png")


# ---------------------------------------------------------------------- CNN
def cnn():
    fig, ax = canvas(9.4, 3.1)
    box(ax, 0.02, 0.44, 0.125, 0.24, "Page image\n128 × 128 × 3", fc="white", bold=True, size=8.5)
    for x, w, h in [(0.195, 0.09, 0.28), (0.325, 0.085, 0.23), (0.45, 0.08, 0.18)]:
        box(ax, x, 0.56 - h / 2, w, h, "Conv\n+ ReLU\nPool", fc=MID, size=7.5)
    ax.text(0.3425, 0.27, "MobileNetV2 convolutional base — pretrained on ImageNet, frozen",
            ha="center", va="top", fontsize=8.5, color=GREY, style="italic")
    box(ax, 0.575, 0.46, 0.115, 0.20, "Global\naverage\npooling", size=8)
    box(ax, 0.715, 0.46, 0.105, 0.20, "Dense 64\n+ ReLU\nDropout 0.4", size=7.5)
    box(ax, 0.845, 0.46, 0.135, 0.20, "Dense 1\n+ Sigmoid", fc=NAVY, ec=NAVY, tc="white",
        bold=True, size=8.5)
    ax.text(0.9125, 0.41, "P(page has a diagram)", ha="center", va="top", fontsize=8, color=GREY)
    for a, b in [(0.145, 0.195), (0.285, 0.325), (0.41, 0.45), (0.53, 0.575), (0.69, 0.715),
                 (0.82, 0.845)]:
        arrow(ax, (a, 0.56), (b, 0.56))
    title(ax, "Model 4 — Diagram CNN (transfer learning)")
    save(fig, "cnn.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pipeline()
    crossencoder()
    ann()
    cnn()
