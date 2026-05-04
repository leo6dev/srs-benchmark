import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

np.random.seed(42)
rng = np.random.default_rng(42)

BG   = "#FAFAF8"; DARK = "#1a1a2e"
C1   = "#281C59"; C2 = "#4E8D9C"; C3 = "#85C79A"
BLUE = "#2471A3"; RED = "#C1392B"; GREY = "#7f8c8d"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "text.color": DARK,
})

SIZE = 6
BOTTOM_COL = 2.5

def road_centre(row, top_col):
    t = row / (SIZE - 1)
    return top_col * (1 - t) + BOTTOM_COL * t

def make_curved_road(top_col, noise_std=0.015, rng_=None):
    img = np.zeros((SIZE, SIZE))
    for row in range(SIZE):
        cx = road_centre(row, top_col)
        for col in range(SIZE):
            dist = abs(col - cx)
            if dist < 0.5:
                img[row, col] = 1.0
            elif dist < 1.0:
                img[row, col] = 0.6 * (1.0 - (dist - 0.5) / 0.5)
    if rng_ is not None and noise_std > 0:
        img += rng_.normal(0, noise_std, (SIZE, SIZE))
    return np.clip(img, 0, 1)

TOP_COLS = [0, 1, 2.5, 4, 5]
TARGETS  = [0.05, 0.25, 0.5, 0.75, 0.95]
LABELS   = ["Far Left\n$\\hat{y} \\to 0.05$",
            "Left\n$\\hat{y} \\to 0.25$",
            "Straight\n$\\hat{y} \\to 0.50$",
            "Right\n$\\hat{y} \\to 0.75$",
            "Far Right\n$\\hat{y} \\to 0.95$"]
LCOLS = [C1, C2, C3, "#E67E22", RED]

# one clean sample per class (no noise, for display)
clean_imgs = [make_curved_road(tc, noise_std=0.0) for tc in TOP_COLS]
# one noisy sample per class
noisy_imgs = [make_curved_road(tc, noise_std=0.015, rng_=rng) for tc in TOP_COLS]

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.patch.set_facecolor(BG)

for j, (tc, label, colour, cimg, nimg) in enumerate(
        zip(TOP_COLS, LABELS, LCOLS, clean_imgs, noisy_imgs)):

    for row, img in enumerate([cimg, nimg]):
        ax = axes[row, j]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

        # annotate each cell with its value
        for r in range(SIZE):
            for c in range(SIZE):
                val = img[r, c]
                fc = "white" if val < 0.50 else DARK
                ax.text(c, r, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=7.5, color=fc, fontweight="bold")

        for k in range(7):
            ax.axhline(k-0.5, color=DARK, lw=0.5, alpha=0.35)
            ax.axvline(k-0.5, color=DARK, lw=0.5, alpha=0.35)
        ax.set_xticks([]); ax.set_yticks([])

        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(2.5)

        if row == 0:
            ax.set_title(label, fontsize=9.5, color=colour,
                         fontweight="bold", pad=6)

        # Blue arrow showing curve direction (row=0 only)
        if row == 0:
            ax.annotate("",
                xy=(tc, 0.3), xytext=(BOTTOM_COL, SIZE - 1.3),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=BLUE,
                    lw=2.2,
                    mutation_scale=14,
                ))

    # row labels on the left
    if j == 0:
        axes[0, 0].set_ylabel("Clean\n(no noise)", fontsize=9, color=DARK,
                               labelpad=6, fontweight="bold")
        axes[1, 0].set_ylabel("Training sample\n(noise $\\sigma$=0.015)",
                               fontsize=9, color=DARK, labelpad=6, fontweight="bold")

plt.tight_layout(pad=0.5)
plt.savefig("/mnt/user-data/outputs/alv3_fig1_v2.png", dpi=180, bbox_inches="tight")
plt.close()
print("fig1 saved")