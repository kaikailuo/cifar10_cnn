"""Vector diagram of src.models.VGGNet; no training or model inference.

Run: python report/draw_architecture.py
Slabs represent channel groups schematically, not activation maps.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "figures"
INK, BLUE, GREEN = "#263F50", "#567E98", "#408572"
PALE, GRID = "#E7EFF4", "#C4D4DE"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig = plt.figure(figsize=(6.3, 3.1), facecolor="white")
ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 16), ylim=(0, 7.9))
ax.axis("off")

def text(x, y, value, size=9, **kw):
    return ax.text(x, y, value, ha=kw.pop("ha", "center"), va="center",
                   fontsize=size, color=INK, **kw)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={"arrowstyle": "->", "color": BLUE,
                            "lw": 1.0, "shrinkA": 0, "shrinkB": 0})

def planes(x, y, side, count):
    # Frontal planes with a small 2-D offset, no perspective or shading.
    for i in reversed(range(count)):
        ax.add_patch(Rectangle((x + i * .075, y + i * .055), side, side,
                               facecolor=PALE, edgecolor=BLUE, lw=.65))
    for f in (.25, .5, .75):
        ax.plot([x + side * f] * 2, [y, y + side], color=GRID, lw=.4)
        ax.plot([x, x + side], [y + side * f] * 2, color=GRID, lw=.4)

def vector(x, height, count, color=BLUE):
    for i in range(count):
        yy = 5.55 - height / 2 + height * i / (count - 1)
        ax.plot([x, x + .33], [yy, yy], color=color, lw=1.2)

# A continuous left-to-right path: feature tensors -> vector -> classifier.
text(4.5, 7.55, "CONVOLUTIONAL FEATURES", 9, weight="bold")
text(12.45, 7.55, "CLASSIFIER", 9, weight="bold")
for x, side, count, label, dims in [
    (.25, 1.6, 3, "Input", "3 × 32 × 32"),
    (3.15, 1.12, 5, "Stage 1", "64 × 16 × 16"),
    (6.0, .78, 7, "Stage 2", "128 × 8 × 8"),
    (8.2, .5, 9, "Stage 3", "256 × 4 × 4"),
]:
    planes(x, 5.55 - side / 2, side, count)
    cx = x + (side + (count - 1) * .075) / 2
    text(cx, 6.95, label, 9.5)
    text(cx, 4.25, dims, 9)
for a, b in [(2.15, 3.02), (4.7, 5.87), (7.35, 8.07)]:
    arrow(a, 5.55, b, 5.55)
vector(10.6, 1.6, 14)
vector(12.6, 1.08, 9, GREEN)
vector(15.1, 1.35, 10)
arrow(9.45, 5.55, 10.45, 5.55)
arrow(11.08, 5.55, 12.43, 5.55)
arrow(13.08, 5.55, 13.4, 5.55)
arrow(14.58, 5.55, 14.95, 5.55)
text(10.76, 6.95, "Flatten", 9.5)
text(10.76, 4.25, "4096", 9.5)
text(12.76, 6.95, "FC + ReLU", 9.5)
text(12.76, 4.25, "256", 9.5)
text(15.1, 6.95, "FC output", 9.5)
text(15.25, 4.25, "10 logits", 9.5)
# Hidden dropout is between the hidden ReLU and final Linear.
ax.add_patch(Rectangle((13.45, 5.1), 1.08, .9, facecolor="white",
                       edgecolor=GREEN, ls="--", lw=.85))
text(13.99, 5.55, "Dropout", 8)

# Local expansion: exact insertion order without text inside feature maps.
ax.plot([.2, 15.8], [3.62, 3.62], color=GRID, lw=.7)
text(.25, 3.08, "Inside each stage", 9.5, ha="left", weight="bold")
cy = 1.92
for x, w, name, optional in [
    (.25, 1.75, "3×3 Conv", False),
    (2.4, 1.35, "Norm", True),
    (4.15, 1.35, "ReLU", False),
    (7.3, 2.45, "2×2 MaxPool", False),
    (11.0, 2.45, "Dropout2d", True),
]:
    ax.add_patch(Rectangle((x, cy - .36), w, .72,
                           facecolor="white" if optional else PALE,
                           edgecolor=GREEN if optional else BLUE,
                           ls="--" if optional else "-", lw=.8))
    text(x + w / 2, cy, name, 9)
for a, b in [(2.05, 2.35), (3.8, 4.1), (5.55, 7.2), (9.8, 10.9)]:
    arrow(a, cy, b, cy)
ax.plot([.25, .25, 5.5, 5.5], [1.35, 1.15, 1.15, 1.35], color=BLUE, lw=.7)
text(2.875, .72, "repeat ×2 · stride 1 · padding 1", 8.5)
text(8.525, .98, "stride 2", 8.5)
text(12.225, .98, "channel dropout", 8.5)
text(15.7, 2.87, "Dashed: optional", 8.5, ha="right")
text(15.7, .32, "Baseline: no Norm or Dropout", 8.5, ha="right")

OUT.mkdir(exist_ok=True)
for extension in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"architecture.{extension}", dpi=200, facecolor="white")
plt.close(fig)
print("Wrote architecture.pdf / architecture.svg / PNG preview.")
