import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

np.random.seed(42)
rng = np.random.default_rng(42)

BG   = "#FAFAF8"; DARK = "#1a1a2e"
C1   = "#281C59"; C2 = "#4E8D9C"; C3 = "#85C79A"; C4 = "#EDF7BD"
RED  = "#C1392B"; BLUE = "#2471A3"; GREY = "#7f8c8d"; GREEN = "#1E8449"

# Custom diverging colormap using the palette
CMAP_W = LinearSegmentedColormap.from_list(
    "alvinn", [C1, C2, C4], N=512
)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "text.color": DARK, "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK,
})

SIZE = 6
BOTTOM_COL = 2.5  # road always starts between cols 2-3 (centre of 6-wide grid)

# ─────────────────────────────────────────────────────────────────────────────
# ROAD GENERATION  — very sharp, low noise
# ─────────────────────────────────────────────────────────────────────────────

def road_centre(row, top_col, bottom_col=BOTTOM_COL, size=SIZE):
    """Linear interpolation: row SIZE-1 = bottom, row 0 = top"""
    t = row / (size - 1)     # 0=top, 1=bottom
    return top_col * (1 - t) + bottom_col * t

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
LABELS   = ["Far Left\n(ŷ→0.05)", "Left\n(ŷ→0.25)", "Straight\n(ŷ→0.5)",
            "Right\n(ŷ→0.75)", "Far Right\n(ŷ→0.95)"]
LCOLS    = [C1, C2, C3, "#E67E22", RED]

X_list, y_list = [], []
for tc, tgt in zip(TOP_COLS, TARGETS):
    for _ in range(80):
        X_list.append(make_curved_road(tc, noise_std=0.015, rng_=rng).flatten())
        y_list.append(tgt)

X = np.array(X_list); y = np.array(y_list)
perm = rng.permutation(len(X)); X, y = X[perm], y[perm]

# ─────────────────────────────────────────────────────────────────────────────
# NETWORK
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(z):   return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def dsigmoid(z):  s = sigmoid(z); return s*(1-s)

w = rng.standard_normal(SIZE*SIZE) * 0.05
b = 0.0
lr, mu, epochs = 0.01, 0.9, 1200
vw, vb = np.zeros_like(w), 0.0
losses = []

for ep in range(epochs):
    idx = rng.permutation(len(X))
    ep_loss = 0.0
    for i in idx:
        xi, ti = X[i], y[i]
        z     = w @ xi + b
        yh    = sigmoid(z)
        err   = yh - ti
        ep_loss += err**2
        gz = 2 * err * dsigmoid(z)
        vw = mu*vw - lr*gz*xi; w += vw
        vb = mu*vb - lr*gz;    b += vb
    losses.append(ep_loss / len(X))

y_hat_all = sigmoid(X @ w + b)
mse = np.mean((y_hat_all - y)**2)
print(f"Final MSE: {mse:.6f},  b = {b:.4f}")
print(f"w (6x6):\n{w.reshape(SIZE,SIZE).round(4)}")

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE: far-right curve (top_col=5, target=0.95)
# ─────────────────────────────────────────────────────────────────────────────
x_s      = make_curved_road(top_col=5, noise_std=0.0).flatten()
z_s      = w @ x_s + b
yhat_s   = sigmoid(z_s)
prod_s   = w * x_s
w_img    = w.reshape(SIZE, SIZE)
xi_img   = x_s.reshape(SIZE, SIZE)
prod_img = prod_s.reshape(SIZE, SIZE)

print(f"\n=== SAMPLE: far-right (top_col=5, no noise) ===")
print(f"x:\n{xi_img.round(4)}")
print(f"w·x = {(w@x_s):.4f},  b = {b:.4f},  z = {z_s:.4f}")
print(f"sigma(z) = {yhat_s:.4f}  (target 0.95)")

print("\n=== PER-ROW BREAKDOWN ===")
for r in range(SIZE):
    terms = [(c, w_img[r,c], xi_img[r,c], prod_img[r,c])
             for c in range(SIZE) if xi_img[r,c] > 0.001]
    row_sum = prod_img[r,:].sum()
    term_str = "  +  ".join([f"({w_:+.4f})({x_:.4f})" for _,w_,x_,_ in terms])
    print(f"  row {r+1}: {term_str}  =  {row_sum:+.4f}")
print(f"\n  Σ(w⊙x) = {prod_s.sum():.4f}")
print(f"  b      = {b:.4f}")
print(f"  z      = {z_s:.4f}")
print(f"  σ(z)   = {yhat_s:.6f}  (target 0.95)")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 – Curved road samples
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.patch.set_facecolor(BG)
for j, (tc, label, colour) in enumerate(zip(TOP_COLS, LABELS, LCOLS)):
    imgs = [make_curved_road(tc, rng_=rng) for _ in range(2)]
    for row in range(2):
        ax = axes[row, j]
        ax.imshow(imgs[row], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for k in range(7):
            ax.axhline(k-0.5, color=DARK, lw=0.5, alpha=0.35)
            ax.axvline(k-0.5, color=DARK, lw=0.5, alpha=0.35)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(2.5)
        if row == 0:
            ax.set_title(label, fontsize=8.5, color=colour, fontweight="bold", pad=5)
        if row == 0:
            ax.annotate("", xy=(tc, 0.2), xytext=(BOTTOM_COL, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2, alpha=0.9))
fig.text(0.5, 0.01,
    "Row 6 (bottom) = car's position  ·  Row 1 (top) = horizon  ·  "
    "Yellow arrow = direction of curve",
    ha="center", fontsize=8.5, color=GREY, style="italic")
fig.suptitle(
    "Figure 1 – Synthetic 6×6 curved road images\n"
    "Road always originates at bottom-centre and curves toward the horizon",
    fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout(pad=0.4)
plt.savefig("/mnt/user-data/outputs/alv3_fig1_samples.png", dpi=180, bbox_inches="tight")
plt.close(); print("\nfig1 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 – Learned weight map with road overlays
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                         gridspec_kw={"width_ratios": [1.1, 1]})
vmax  = max(np.abs(w_img).max(), 1e-6)
norm  = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

ax = axes[0]
im = ax.imshow(w_img, cmap=CMAP_W, norm=norm, interpolation="nearest")
for r in range(SIZE):
    for c in range(SIZE):
        val = w_img[r, c]
        # text colour: dark on light cells, light on dark cells
        bg_brightness = (val - (-vmax)) / (2*vmax)   # 0=dark, 1=light
        fc = DARK if bg_brightness > 0.45 else "#EDF7BD"
        ax.text(c, r, f"{val:+.3f}", ha="center", va="center",
                fontsize=8.5, color=fc, fontweight="bold")
for k in range(7):
    ax.axhline(k-0.5, color="#281C59", lw=0.8, alpha=0.4)
    ax.axvline(k-0.5, color="#281C59", lw=0.8, alpha=0.4)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels([f"col {i+1}" for i in range(6)], fontsize=8.5)
ax.set_yticklabels([f"row {i+1}" for i in range(6)], fontsize=8.5)
ax.set_title("Learned weight matrix  w  (reshaped 6×6)\n"
             "Dark purple = large negative  ·  Light yellow = large positive",
             fontsize=9.5, fontweight="bold", pad=8)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("weight value", fontsize=9)

ax2 = axes[1]
im2 = ax2.imshow(w_img, cmap=CMAP_W, norm=norm, interpolation="nearest", alpha=0.85)
road_traces = [(0, C1, "Far Left  (ŷ→0.05)"),
               (2.5, C3, "Straight  (ŷ→0.5)"),
               (5,   RED,"Far Right (ŷ→0.95)")]
for tc, col, lbl in road_traces:
    ys = np.linspace(0, SIZE-1, 60)
    xs = [road_centre(r, tc) for r in ys]
    ax2.plot(xs, ys, color=col, lw=3.5, alpha=0.95, label=lbl,
             linestyle="--" if tc==2.5 else "-")
ax2.set_xticks(range(6)); ax2.set_yticks(range(6))
ax2.set_xticklabels([f"col {i+1}" for i in range(6)], fontsize=8.5)
ax2.set_yticklabels([f"row {i+1}" for i in range(6)], fontsize=8.5)
ax2.legend(fontsize=9, loc="lower right", framealpha=0.92, edgecolor=DARK)
ax2.set_title("Weight matrix with road traces overlaid\n"
              "Left roads cross dark (negative) weights  ·  Right roads cross light (positive)",
              fontsize=9.5, fontweight="bold", pad=8)
for k in range(7):
    ax2.axhline(k-0.5, color="#281C59", lw=0.5, alpha=0.3)
    ax2.axvline(k-0.5, color="#281C59", lw=0.5, alpha=0.3)

fig.suptitle(
    "Figure 2 – Learned weight vector w (36 values, reshaped to 6×6)\n"
    "Top rows are highly discriminative; bottom rows near-zero (road always at centre there)",
    fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout(pad=0.6)
plt.savefig("/mnt/user-data/outputs/alv3_fig2_weights.png", dpi=180, bbox_inches="tight")
plt.close(); print("fig2 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 – Visual forward pass  (simplified, non-technical titles, fixed contrast)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
fig.patch.set_facecolor(BG)

# --- panel 1: input x (grayscale 0-1) ---
ax = axes[0]
ax.imshow(xi_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
for r in range(SIZE):
    for c in range(SIZE):
        val = xi_img[r, c]
        fc = "white" if val < 0.55 else DARK    # dark text on bright, white text on dark
        ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color=fc, fontweight="bold")
for k in range(7):
    ax.axhline(k-0.5, color=DARK, lw=0.6, alpha=0.4)
    ax.axvline(k-0.5, color=DARK, lw=0.6, alpha=0.4)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Input vector  x\n(far-right road image,\nflattened to 36×1)",
             fontsize=10.5, fontweight="bold", pad=8, color=DARK)
# colourbar: 0-1 gradient key
from mpl_toolkits.axes_grid1 import make_axes_locatable
div = make_axes_locatable(ax)
cax = div.append_axes("bottom", size="6%", pad=0.06)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0,1), cmap="gray"),
                  cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0  (dark/grass)", "0.5", "1  (bright/road)"])
cb.ax.tick_params(labelsize=7.5)

# --- panel 2: weight matrix w (custom cmap) ---
ax = axes[1]
vmax2 = max(np.abs(w_img).max(), 1e-6)
norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
ax.imshow(w_img, cmap=CMAP_W, norm=norm2, interpolation="nearest")
for r in range(SIZE):
    for c in range(SIZE):
        val = w_img[r, c]
        bg_brightness = (val + vmax2) / (2*vmax2)    # 0=darkest, 1=lightest
        fc = DARK if bg_brightness > 0.45 else "#EDF7BD"
        ax.text(c, r, f"{val:+.3f}", ha="center", va="center",
                fontsize=9, color=fc, fontweight="bold")
for k in range(7):
    ax.axhline(k-0.5, color="#281C59", lw=0.6, alpha=0.4)
    ax.axvline(k-0.5, color="#281C59", lw=0.6, alpha=0.4)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Weight matrix  w\n(learned, 36×1\nreshaped to 6×6)",
             fontsize=10.5, fontweight="bold", pad=8, color=DARK)
div2 = make_axes_locatable(ax)
cax2 = div2.append_axes("bottom", size="6%", pad=0.06)
cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(-vmax2, vmax2), cmap=CMAP_W),
                   cax=cax2, orientation="horizontal")
cb2.set_ticks([-vmax2, 0, vmax2])
cb2.set_ticklabels([f"{-vmax2:.2f}  (steer left)", "0", f"{vmax2:.2f}  (steer right)"])
cb2.ax.tick_params(labelsize=7.5)

# --- panel 3: element-wise product w⊙x ---
ax = axes[2]
vmax3 = max(np.abs(prod_img).max(), 1e-6)
norm3 = TwoSlopeNorm(vmin=-vmax3, vcenter=0, vmax=vmax3)
ax.imshow(prod_img, cmap=CMAP_W, norm=norm3, interpolation="nearest")
for r in range(SIZE):
    for c in range(SIZE):
        val = prod_img[r, c]
        bg_brightness = (val + vmax3) / (2*vmax3)
        fc = DARK if bg_brightness > 0.45 else "#EDF7BD"
        ax.text(c, r, f"{val:+.3f}", ha="center", va="center",
                fontsize=9, color=fc, fontweight="bold")
for k in range(7):
    ax.axhline(k-0.5, color="#281C59", lw=0.6, alpha=0.4)
    ax.axvline(k-0.5, color="#281C59", lw=0.6, alpha=0.4)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Product  w ⊙ x\n(entry-wise multiply)\n"
             f"z = Σ(w⊙x) + b = {z_s:.3f}",
             fontsize=10.5, fontweight="bold", pad=8, color=DARK)
div3 = make_axes_locatable(ax)
cax3 = div3.append_axes("bottom", size="6%", pad=0.06)
cb3 = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(-vmax3, vmax3), cmap=CMAP_W),
                   cax=cax3, orientation="horizontal")
cb3.set_ticks([-vmax3, 0, vmax3])
cb3.set_ticklabels([f"{-vmax3:.2f}", "0", f"+{vmax3:.2f}"])
cb3.ax.tick_params(labelsize=7.5)

# result annotation below panel 3
fig.text(0.83, 0.04,
         rf"$\hat{{y}} = \sigma({z_s:.3f}) = {yhat_s:.4f}$   →   steer far right ✓",
         ha="center", fontsize=12, fontweight="bold", color=RED)

fig.suptitle(
    r"Figure 3 – Forward pass:  $\hat{y} = \sigma(\mathbf{w}^\top\mathbf{x}+b)$   "
    r"·   Input: far-right curve (target $\hat{y} \to 0.95$)",
    fontsize=12, fontweight="bold", y=1.02)

plt.tight_layout(pad=0.8)
plt.savefig("/mnt/user-data/outputs/alv3_fig3_forward.png", dpi=180, bbox_inches="tight")
plt.close(); print("fig3 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 – Results
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
pt_colours = [C1, C2, C3, "#E67E22", RED]
for tc, tgt, lbl, col in zip(TOP_COLS, TARGETS, LABELS, pt_colours):
    mask = np.abs(y - tgt) < 0.01
    ax1.scatter(y[mask], y_hat_all[mask], alpha=0.55, s=20, color=col,
                label=lbl.replace("\n"," "), zorder=3)
ax1.plot([0,1],[0,1], color=DARK, lw=1.5, ls="--", label="perfect prediction")
ax1.set_xlabel("Target  y", fontsize=10); ax1.set_ylabel("Predicted  ŷ", fontsize=10)
ax1.set_title("Figure 4a – Predictions vs targets", fontsize=10, fontweight="bold")
ax1.legend(fontsize=7.5); ax1.grid(True, alpha=0.3); ax1.set_xlim(0,1); ax1.set_ylim(0,1)

ax2.plot(np.arange(1,epochs+1), losses, color=C2, lw=2)
ax2.set_xlabel("Epoch", fontsize=10); ax2.set_ylabel("MSE Loss", fontsize=10)
ax2.set_title(f"Figure 4b – Training loss  (final MSE = {mse:.5f})", fontsize=10, fontweight="bold")
ax2.grid(True, alpha=0.3); ax2.set_xlim(1,epochs); ax2.set_ylim(0)
plt.tight_layout(pad=0.5)
plt.savefig("/mnt/user-data/outputs/alv3_fig4_results.png", dpi=180, bbox_inches="tight")
plt.close(); print("fig4 saved")

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