#!/usr/bin/env python3
"""Recompute the documented planning-only split arithmetic (MiB and ms)."""
MiB = 1 << 20
L = 210_166_784 / MiB
S = 185_836_544 / MiB
EMBED = 437_575_680 / MiB
HEAD = 638_131_200 / MiB
KV_PER_LAYER = 134_217_728 / MiB
# 32-row observed max graph temporary allowance, derived from 863.06 MiB graph
# allocation minus 128 MiB KV plus 106 MiB observed post-run workspace.
TEMP = 863.06 - 128 + 106
large = {0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39}
# Modeled usable budgets after baseline device/background allowances and 512 MiB reserve.
BUDGET_3060 = 11_405
BUDGET_2080S = 6_885
# Representative medians in ms by GPU, class (L/S), and rows.
t = {
    0: {"L": [12.187185, 14.797016, 18.207218, 24.372563, 36.422879],
        "S": [12.000964, 14.655366, 17.906829, 24.122951, 36.174417]},
    1: {"L": [4.557817, 6.183143, 8.880004, 13.668612, 24.208701],
        "S": [3.923155, 6.265183, 8.797853, 13.774991, 24.496264]},
}
rows = [1, 16, 32, 64, 128]
# Transfer median ms for direction A (3060 -> 2080S), B reverse.
transfer = {"A": [0.02084, 0.13333, 0.25504, 0.49921, 0.98630],
            "B": [0.02077, 0.15008, 0.28864, 0.56626, 1.12041]}
head = {0: [1.959267, 2.545929, 3.020451, 4.300335, 6.767843],
        1: [1.602716, 2.203717, 2.543688, 3.392111, 6.527112]}
embed = {0: {1: .007700, 32: .011040}, 1: {1: .005870, 32: .008480}}

for cut in range(24, 29):
    early_l = sum(i in large for i in range(cut))
    early_s = cut - early_l
    late_l = sum(i in large for i in range(cut, 40))
    late_s = 40 - cut - late_l
    # Direction A modeled peak usage; each GPU has its own one-layer KV block overhead
    # abstraction used by the prior planning estimate.
    early = early_l * L + early_s * S + EMBED + cut * KV_PER_LAYER + TEMP
    late = late_l * L + late_s * S + HEAD + (40-cut) * KV_PER_LAYER + TEMP
    print(f"cut={cut}/ {40-cut} counts={early_l}/{early_s}+{late_l}/{late_s} "
          f"A_headroom_MiB={BUDGET_3060-early:.2f},{BUDGET_2080S-late:.2f} "
          f"A_headroom_after_extra_512={BUDGET_3060-early-512:.2f},{BUDGET_2080S-late-512:.2f}")
    for direction, early_gpu, late_gpu in (("A", 0, 1), ("B", 1, 0)):
        for ri, n in enumerate(rows):
            block_ms = sum(t[early_gpu]["L" if i in large else "S"][ri] for i in range(cut))
            block_ms += sum(t[late_gpu]["L" if i in large else "S"][ri] for i in range(cut, 40))
            total = block_ms + transfer[direction][ri] + head[late_gpu][ri]
            if n in embed[early_gpu]:
                total += embed[early_gpu][n]
            print(f"  {direction} rows={n} projected_ms={total:.3f}")
