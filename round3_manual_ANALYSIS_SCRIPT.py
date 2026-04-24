"""
Trading Game Optimizer (vectorized)
===================================
Finds optimal First Bid (b1) and Second Bid (b2) to maximise Total PnL.

Game rules
----------
  51 counterparties with reserve prices r ∈ {670, 675, ..., 920}.
  Fair (sell) price = 920.

  For each counterparty with reserve r:
    a) b1 > r                           → Bid1 trade:  PnL per trade = 920 - b1
    b) b1 ≤ r, b2 > r, b2 > field_avg  → AutoTrade:   PnL per trade = 920 - b2
    c) b1 ≤ r, b2 > r, b2 ≤ field_avg  → NonAuto:     PnL per trade = ((920-field_avg)/(920-b2))^3
    d) otherwise (b2 ≤ r)               → no trade, PnL = 0
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ─────────────────────────────────────────────────────────────────
FAIR = 920
RES  = np.arange(670, 921, 5)   # 51 reserve prices
BIDS = np.arange(670, 921, 1)   # 51 valid bid values
N    = len(RES)

field_avg = 800   # <── change this and re-run


# ── Vectorized PnL grid ────────────────────────────────────────────────────────
def pnl_grid(fa: float) -> np.ndarray:
    """
    Return a (len(BIDS), len(BIDS)) array where grid[j, i] is the total PnL
    for b1 = BIDS[i], b2 = BIDS[j]. Fully vectorized over both bids and reserves.
    """
    b1 = BIDS[None, :, None]        # shape (1, n_b1, 1)
    b2 = BIDS[:, None, None]        # shape (n_b2, 1, 1)
    r  = RES[None, None, :]         # shape (1, 1, n_res)

    # Bid1 component: depends only on b1
    bid1_mask = b1 > r                                   # (1, n_b1, n_res)
    bid1_pnl  = (bid1_mask * (FAIR - b1)).sum(axis=-1)   # (1, n_b1)

    # Auto component: needs b1 ≤ r, b2 > r, b2 > fa
    auto_mask = (~bid1_mask) & (b2 > r) & (b2 > fa)      # (n_b2, n_b1, n_res)
    auto_pnl  = (auto_mask * (FAIR - b2)).sum(axis=-1)   # (n_b2, n_b1)

    # Decay component: b1 ≤ r, b2 > r, b2 ≤ fa  (b2 ≤ r → no trade, contributes 0)
    nonauto_mask = (~bid1_mask) & (b2 > r) & (b2 <= fa)  # (n_b2, n_b1, n_res)
    nonauto = nonauto_mask.sum(axis=-1)                  # (n_b2, n_b1)
    # Guard b2 == FAIR (decay undefined) → contribute 0
    b2_flat = BIDS[:, None].astype(float)                # (n_b2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay_factor = np.where(
            b2_flat < FAIR,
            ((FAIR - fa) / (FAIR - b2_flat)) ** 3,
            0.0,
        )                                                # (n_b2, 1)
    decay_pnl = nonauto * decay_factor                   # (n_b2, n_b1)

    return bid1_pnl + auto_pnl + decay_pnl               # (n_b2, n_b1)


def best_bids(fa: float):
    """Return (best_b1, best_b2, best_pnl) for a given field average."""
    g = pnl_grid(fa)
    j, i = np.unravel_index(np.argmax(g), g.shape)
    return int(BIDS[i]), int(BIDS[j]), float(g[j, i])


# ── Robust optimization under uncertainty about field_avg ──────────────────────
def expected_pnl_grid(fa_dist: dict) -> np.ndarray:
    """
    fa_dist: {fa_value: probability}. Probabilities need not sum to 1;
    they will be normalized. Returns expected PnL grid.
    """
    total_p = sum(fa_dist.values())
    grid = np.zeros((len(BIDS), len(BIDS)))
    for fa, p in fa_dist.items():
        grid += (p / total_p) * pnl_grid(fa)
    return grid


def best_expected_bids(fa_dist: dict):
    g = expected_pnl_grid(fa_dist)
    j, i = np.unravel_index(np.argmax(g), g.shape)
    return int(BIDS[i]), int(BIDS[j]), float(g[j, i])


def worst_case_bids(fa_values):
    """Maximin: pick (b1, b2) maximizing the worst PnL across fa_values."""
    stack = np.stack([pnl_grid(fa) for fa in fa_values])   # (n_fa, n_b2, n_b1)
    min_grid = stack.min(axis=0)
    j, i = np.unravel_index(np.argmax(min_grid), min_grid.shape)
    return int(BIDS[i]), int(BIDS[j]), float(min_grid[j, i])


def evaluate_strategy(b1, b2, fa_values):
    """Return array of PnLs for a given (b1, b2) across fa_values."""
    i = int(np.where(BIDS == b1)[0][0])
    j = int(np.where(BIDS == b2)[0][0])
    return np.array([pnl_grid(fa)[j, i] for fa in fa_values])


# ── Cliff / marginal analysis ─────────────────────────────────────────────────
def pnl_scalar(b1, b2, fa):
    """Scalar PnL for a single (b1, b2, fa). Used for reports."""
    n1 = int(np.sum(b1 > RES))
    bid1_p = n1 * (FAIR - b1)
    if b2 >= FAIR:
        return float(bid1_p)
    n_cap = int(np.sum((b1 <= RES) & (b2 > RES)))
    if b2 > fa:
        return float(bid1_p + n_cap * (FAIR - b2))
    return float(bid1_p + n_cap * ((FAIR - fa) / (FAIR - b2)) ** 3)


def n_cap_of(b1, b2):
    return int(np.sum((b1 <= RES) & (b2 > RES)))


def strategy_report(fa, deltas=(-10, -5, -1, 0, 1, 5, 10, 15)):
    """
    Full report for field_avg = fa:
      - optimum (b1*, b2*) and PnL
      - effect of deviating b1 and b2 individually
      - cliff profile: how PnL changes as fa varies around the optimum
      - insurance cost/benefit table
    """
    b1s, b2s, opt = best_bids(fa)
    ncap = n_cap_of(b1s, b2s)
    cliff = ncap * (FAIR - b2s - 1)

    print(f"\n── fa = {fa} " + "─" * 52)
    print(f"  Optimum       : b1 = {b1s}, b2 = {b2s}   PnL = {opt:.1f}")
    print(f"  N_cap         : {ncap}  (reserves in [b1, b2))")
    print(f"  Cliff height  : {cliff:.0f}   (drop if fa reaches b2)")

    print(f"  b2 deviations (b1 = {b1s} fixed, fa = {fa}):")
    for d in deltas:
        b2_new = b2s + d
        if b2_new < BIDS[0] or b2_new > BIDS[-1]:
            continue
        p = pnl_scalar(b1s, b2_new, fa)
        tag = "CLIFF ↓" if b2_new <= fa else "        "
        print(f"    b2={b2_new:<4} (Δ={d:+3d}): PnL = {p:>7.1f}  ({p-opt:+7.1f})  {tag}")

    print(f"  b1 deviations (b2 = {b2s} fixed, fa = {fa}):")
    for d in deltas:
        b1_new = b1s + d
        if b1_new < BIDS[0] or b1_new > BIDS[-1]:
            continue
        p = pnl_scalar(b1_new, b2s, fa)
        print(f"    b1={b1_new:<4} (Δ={d:+3d}): PnL = {p:>7.1f}  ({p-opt:+7.1f})")

    print(f"  Cliff profile (strategy fixed at optimum, fa varies):")
    for fa_new in sorted({fa - 5, fa - 1, fa, fa + 1, fa + 2, fa + 5, fa + 10}):
        p = pnl_scalar(b1s, b2s, fa_new)
        tag = "CLIFF ↓" if b2s <= fa_new else "        "
        print(f"    fa={fa_new:<4}  PnL = {p:>7.1f}  ({p-opt:+7.1f})  {tag}")


def insurance_table(fa_target, insurance_ks=(0, 2, 5, 10, 15, 20)):
    """
    For fa = fa_target, show the cost / cliff tradeoff of bidding b2 higher
    than the point optimum. Each row shows:
      - b2 = (fa_target + 1 + k): bid b2 this high
      - PnL if fa = fa_target (insurance cost = reduction vs k=0)
      - fa_max protected: largest fa this strategy still auto-trades against
      - cliff drop if fa exceeds b2: what you lose if you under-insured
      - breakeven p: min probability of fa > b2_base needed for this k to pay off
    """
    b1s, _, _ = best_bids(fa_target)
    base_pnl = pnl_scalar(b1s, fa_target + 1, fa_target) if fa_target + 1 in BIDS else None

    print(f"\nINSURANCE TABLE  (fa_target = {fa_target}, b1 = {b1s})")
    print(f"  {'k':>3}  {'b2':>4}  {'PnL@fa*':>9}  {'cost':>6}  {'fa_max':>6}  {'cliff':>6}  {'brk%':>5}")
    for k in insurance_ks:
        b2 = fa_target + 1 + k
        if b2 not in BIDS:
            continue
        pnl_here = pnl_scalar(b1s, b2, fa_target)
        cost = (base_pnl - pnl_here) if base_pnl is not None else 0
        cliff = n_cap_of(b1s, b2) * (FAIR - b2 - 1)
        brk = (cost / cliff * 100) if cliff > 0 else float("nan")
        print(f"  {k:>3}  {b2:>4}  {pnl_here:>9.1f}  {cost:>6.1f}  "
              f"{b2-1:>6}  {cliff:>6.0f}  {brk:>5.1f}")


# ── Validation ─────────────────────────────────────────────────────────────────
# NOTE: the original validation used b1=80, which is outside the bid grid.
# We instead validate the formula directly on the scalar case.
def _scalar_pnl(b1, b2, fa):
    bid1_t = int(np.sum(b1 > RES))
    bid1_p = bid1_t * (FAIR - b1)
    if b2 >= FAIR:
        return bid1_t, bid1_p, 0, 0, 0, 0.0
    auto_t = int(np.sum((b1 <= RES) & (b2 > RES) & (b2 > fa)))
    auto_p = auto_t * (FAIR - b2)
    nona   = int(np.sum((b1 <= RES) & (b2 > RES) & (b2 <= fa)))
    decay  = ((FAIR - fa) / (FAIR - b2)) ** 3
    return bid1_t, bid1_p, auto_t, auto_p, nona, nona * decay

print("=" * 56)
print("VALIDATION CHECK  (b1=80, b2=820, field_avg=800)")
print("=" * 56)
bt, bp, at, ap, na, dp = _scalar_pnl(80, 820, 800)
print(f"  Bid1_Trades     = {bt:>4}    expected   0")
print(f"  Bid1_PnL        = {bp:>4}    expected   0")
print(f"  Bid2_AutoTrades = {at:>4}    expected  30")
print(f"  Bid2_Auto_PnL   = {ap:>4}    expected 3000")
print(f"  Bid2_NonAutos   = {na:>4}    expected   0")
print(f"  Bid2_Decay_PnL  = {dp:>8.3f}    expected   0.000")
print(f"  Total_PnL       = {bp+ap+dp:>8.3f}    expected 3000.000")
print()


# ── Brute-force search (now a single numpy expression) ─────────────────────────
print("=" * 56)
print(f"BRUTE-FORCE SEARCH  (field_avg = {field_avg})")
print("=" * 56)
grid = pnl_grid(field_avg)
best_b1, best_b2, best_p = best_bids(field_avg)
print(f"  Optimal b1  = {best_b1}")
print(f"  Optimal b2  = {best_b2}")
print(f"  Maximum PnL = {best_p:.3f}")
print()


# ── Heatmap ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
half = (BIDS[1] - BIDS[0]) / 2   # half a bid step, for pixel centering
im = ax.imshow(
    grid, origin="lower", aspect="auto", cmap="RdYlGn",
    extent=[BIDS[0] - half, BIDS[-1] + half, BIDS[0] - half, BIDS[-1] + half],
)
plt.colorbar(im, ax=ax, label="Total PnL")
ticks = list(range(670, 921, 50))
ax.set_xticks(ticks); ax.set_yticks(ticks)
ax.set_xlabel("First Bid  (b1)", fontsize=12)
ax.set_ylabel("Second Bid  (b2)", fontsize=12)
ax.set_title(
    f"Total PnL Heatmap  —  field_avg = {field_avg}\n"
    f"Optimal: b1 = {best_b1},  b2 = {best_b2},  PnL = {best_p:.1f}",
    fontsize=13,
)
ax.axvline(best_b1, color="white", lw=1.5, ls="--", alpha=0.85, label=f"opt b1 = {best_b1}")
ax.axhline(best_b2, color="cyan",  lw=1.5, ls="--", alpha=0.85, label=f"opt b2 = {best_b2}")
ax.legend(fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig("pnl_heatmap.png", dpi=150)
plt.close(fig)
print("Heatmap saved to  pnl_heatmap.png")
print()


# ── Sensitivity table ──────────────────────────────────────────────────────────
print("=" * 56)
print("SENSITIVITY TABLE  (field_avg: 670 → 920, step 10)")
print("=" * 56)
print(f"{'field_avg':>10}  {'opt_b1':>7}  {'opt_b2':>7}  {'Total_PnL':>12}")
print("-" * 44)
for fa in range(670, 921, 10):
    bb1, bb2, bp2 = best_bids(fa)
    print(f"{fa:>10}  {bb1:>7}  {bb2:>7}  {bp2:>12.3f}")


# ── Robust analysis under uncertainty about field_avg ──────────────────────────
print()
print("=" * 72)
print("ROBUST ANALYSIS  — optimizing under uncertainty about field_avg")
print("=" * 72)

# Three belief scenarios about what fa will be:
scenarios = {
    "Narrow (level-1/2 thinkers dominate)":
        {830: 0.2, 835: 0.3, 840: 0.3, 845: 0.15, 850: 0.05},
    "Wider (level-k spread + asymmetric risk pushes up)":
        {820: 0.05, 830: 0.10, 840: 0.20, 850: 0.25, 860: 0.20, 870: 0.12, 880: 0.08},
    "User's prior (20% safe at low-fa, 80% clustered mid)":
        {795: 0.20, 835: 0.25, 845: 0.25, 855: 0.20, 865: 0.10},
}

# Safe baseline: b1=791 bid1-only strategy gives 3225 regardless of fa.
SAFE_PNL = 25 * (FAIR - 791)

for name, dist in scenarios.items():
    b1_r, b2_r, ep = best_expected_bids(dist)
    # Compare to "naively play point estimate of fa"
    point_fa = round(sum(fa * p for fa, p in dist.items()) / sum(dist.values()))
    b1_n, b2_n, _ = best_bids(point_fa)
    # How does the robust strategy perform in the worst case?
    pnls = evaluate_strategy(b1_r, b2_r, list(dist.keys()))
    print(f"\n  {name}")
    print(f"    mean fa              = {point_fa}")
    print(f"    robust (b1, b2)      = ({b1_r}, {b2_r})   expected PnL = {ep:.1f}")
    print(f"    naive  (b1, b2)      = ({b1_n}, {b2_n})   (best response to mean fa)")
    print(f"    worst-case PnL       = {pnls.min():.1f}   (vs safe floor {SAFE_PNL})")
    print(f"    best-case PnL        = {pnls.max():.1f}")

# Maximin across a plausible fa range
print()
fa_range = list(range(820, 881, 5))
b1_w, b2_w, worst = worst_case_bids(fa_range)
print(f"  Maximin over fa ∈ [820, 880]:  (b1, b2) = ({b1_w}, {b2_w}),  min PnL = {worst:.1f}")


# ── Marginal deviation analysis per fa target ─────────────────────────────────
print()
print("=" * 72)
print("MARGINAL DEVIATION ANALYSIS  (per-fa optimum + local tradeoffs)")
print("=" * 72)
for fa_target in [830, 846, 850, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871]:
    strategy_report(fa_target)

print()
print("=" * 72)
print("INSURANCE COST / BENEFIT  (how high to bid b2 above fa*)")
print("=" * 72)
for fa_target in [830, 850, 870]:
    insurance_table(fa_target)


# ── PnL-vs-fa curves for candidate strategies ─────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
fa_sweep = np.arange(800, 901, 1)
candidates = [(751, 836), (771, 875), (771, 885), (791, 670)]
for b1, b2 in candidates:
    curve = [pnl_scalar(b1, b2, f) for f in fa_sweep]
    ax2.plot(fa_sweep, curve, lw=2, label=f"b1={b1}, b2={b2}")
ax2.axhline(3225, color="gray", lw=1, ls=":", label="safe floor (b1=791)")
ax2.set_xlabel("Actual field_avg")
ax2.set_ylabel("PnL")
ax2.set_title("PnL vs actual field_avg for candidate strategies (shows the cliff)")
ax2.legend(loc="lower left")
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cliff_curves.png", dpi=150)
plt.close(fig2)
print("\nCliff curves saved to  cliff_curves.png")
