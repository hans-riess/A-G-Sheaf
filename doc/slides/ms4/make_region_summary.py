"""Generate figures/region_summary.png for the MS4 briefing deck.

Two agents pin *different* tiles of region r unsafe, yet push to the *same*
region-level interface summary: the kernel of the summary map, drawn.
Palette matches the demonstration world (exp/defaults.yaml).
Run with the project venv:  ../../../.venv/bin/python make_region_summary.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

NAVY = "#051E39"
GOLD = "#EAAA00"
UNSAFE = "#D90368"
GREEN = "#066034"
MUTED = "#6b7684"
GRID = "#c9cdd4"

ROWS, COLS = 3, 3
TILE = 0.62
W, H = 11.6, 4.7

fig, ax = plt.subplots(figsize=(W, H), dpi=220)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

CY = 2.62  # vertical center of grids, card, and arrows


def draw_grid(x0, pinned, tile_label, agent, sub):
    """One region drawn as a tile grid with a single unsafe-pinned tile."""
    w, h = COLS * TILE, ROWS * TILE
    y0 = CY - h / 2
    for r in range(ROWS):
        for c in range(COLS):
            ax.add_patch(Rectangle((x0 + c * TILE, y0 + r * TILE), TILE, TILE,
                                   facecolor="white", edgecolor=GRID,
                                   linewidth=0.8))
    pr, pc = pinned
    ax.add_patch(Rectangle((x0 + pc * TILE, y0 + pr * TILE), TILE, TILE,
                           facecolor=UNSAFE, alpha=0.55, edgecolor=UNSAFE,
                           linewidth=1.4))
    ax.text(x0 + (pc + 0.5) * TILE, y0 + (pr + 0.5) * TILE, tile_label,
            ha="center", va="center", fontsize=10, color=NAVY, weight="bold")
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none",
                           edgecolor=NAVY, linewidth=2.2))
    ax.text(x0 + w / 2, CY, "$r$", ha="center", va="center",
            fontsize=26, color=MUTED, alpha=0.40, style="italic")
    ax.text(x0 + w / 2, y0 + h + 0.14, agent, ha="center", va="bottom",
            fontsize=12.5, color=NAVY, weight="bold")
    ax.text(x0 + w / 2, y0 - 0.14, sub, ha="center", va="top",
            fontsize=10, color=MUTED)
    return x0, x0 + w


GW = COLS * TILE
li, ri = draw_grid(0.45, (2, 1), r"$\mathsf{q}_2$",
                   "Agent $i$", r"pins $\mathsf{q}_2$ unsafe")
lj, rj = draw_grid(W - 0.45 - GW, (0, 2), r"$\mathsf{q}_4$",
                   "Agent $j$", r"pins $\mathsf{q}_4$ unsafe")

# Interface card
CW, CH = 3.40, 2.68
cx = W / 2
card = FancyBboxPatch((cx - CW / 2, CY - CH / 2), CW, CH,
                      boxstyle="round,pad=0.10,rounding_size=0.10",
                      facecolor="#f7f8fa", edgecolor=NAVY, linewidth=2.0)
ax.add_patch(card)
ax.text(cx, CY + CH / 2 - 0.26, r"interface $Y_{ij}$ — region $r$",
        ha="center", va="center", fontsize=12, color=NAVY, weight="bold")
entries = [
    (r"$F_{\mathsf{unsafe},\,r} = \top$", "flagged: hazard somewhere in $r$"),
    (r"$X_{\mathsf{target},\,r} = \top$", "excluded: no target anywhere in $r$"),
    (r"$K_{i,r},\ K_{j,r}$", "route claims on $r$"),
]
x_text = cx - CW / 2 + 0.30
for k, (formula, gloss) in enumerate(entries):
    y = CY + 0.50 - k * 0.72
    ax.text(x_text, y, formula, ha="left", va="center",
            fontsize=12.5, color=NAVY)
    ax.text(x_text + 0.22, y - 0.28, gloss, ha="left", va="center",
            fontsize=8.5, color=MUTED, style="italic")

# Pushforward arrows, both landing on the same card
for x_from, x_to, label in [
        (ri + 0.14, cx - CW / 2 - 0.24, r"$(R_{i\,\trianglelefteq\,ij})_{!}$"),
        (lj - 0.14, cx + CW / 2 + 0.24, r"$(R_{j\,\trianglelefteq\,ij})_{!}$")]:
    ax.add_patch(FancyArrowPatch((x_from, CY), (x_to, CY),
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.8, color=NAVY))
    ax.text((x_from + x_to) / 2, CY + 0.16, label, ha="center", va="bottom",
            fontsize=11.5, color=NAVY)

# The claim
ax.text(cx, 0.72, "same summary  $\\Rightarrow$  the interface agrees: the section closes",
        ha="center", va="center", fontsize=12.5, color=GREEN, weight="bold")
ax.text(cx, 0.34, "…over the tile-level disagreement ($\\mathsf{q}_2$ vs. $\\mathsf{q}_4$):"
        " which tile is the kernel the interface cannot express",
        ha="center", va="center", fontsize=10.5, color=MUTED)

fig.savefig("figures/region_summary.png", bbox_inches="tight",
            facecolor="white", pad_inches=0.12)
print("wrote figures/region_summary.png")
