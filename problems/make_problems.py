# 2D grid Ising instances (SAFE ma-QAOA, https://arxiv.org/abs/2605.23377, Sec. III-A): H = sum_i h_i Z_i + sum_(i,j) J_ij Z_i Z_j
# NN ~ N(0, 1), NNN (both diagonals) ~ N(0, 0.3^2), h ~ N(0, 0.1^2), open boundary, seed = 120490 + inst.
# Writes <name>.json (with exhaustive E_min / E_max) and <name>.png next to this script.
# Re-running is a reproducibility check: an existing JSON must come out identical.
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INSTANCES = [(5, 5, 0), (5, 5, 1), (5, 5, 2), (5, 5, 3), (5, 5, 4),
             (3, 3, 0), (3, 3, 1), (3, 3, 2), (3, 3, 3), (3, 3, 4)]  # (rows, cols, inst)
SEED_BASE = 120490
NN_STD, NNN_STD, FIELD_STD = 1.0, 0.3, 0.1
OUT_DIR = Path(__file__).resolve().parent

for rows, cols, inst in INSTANCES:
    seed = SEED_BASE + inst
    n = rows * cols
    name = f"grid{rows}x{cols}_inst{inst}_seed{seed}"

    # ---- problem: same pair order and RNG draw order as the paper generator (NN, then NNN, then fields)
    nn_pairs, nnn_pairs = set(), set()
    for i in range(rows):
        for j in range(cols):
            u = i * cols + j
            if j + 1 < cols:
                nn_pairs.add((u, i * cols + j + 1))
            if i + 1 < rows:
                nn_pairs.add((u, (i + 1) * cols + j))
            if i + 1 < rows and j + 1 < cols:
                nnn_pairs.add((u, (i + 1) * cols + j + 1))
            if i + 1 < rows and j - 1 >= 0:
                nnn_pairs.add((u, (i + 1) * cols + j - 1))
    rng = np.random.default_rng(seed)
    edges = [[u, v, float(rng.normal(0.0, NN_STD))] for (u, v) in sorted(nn_pairs)]
    edges += [[u, v, float(rng.normal(0.0, NNN_STD))] for (u, v) in sorted(nnn_pairs)]
    fields = [[q, float(rng.normal(0.0, FIELD_STD))] for q in range(n)]

    # ---- brute-force solution over all 2^n spin configurations (s = 1 - 2*bit, Z|0> = +1)
    # E_min / E_max normalise the approximation ratio; E_min and its bitstring are the reference line in figures.
    states = np.arange(1 << n, dtype=np.uint32)
    energy = np.zeros(1 << n, dtype=np.float64)
    for u, v, w in edges:
        energy += w * (1.0 - 2.0 * ((states >> u) & 1)) * (1.0 - 2.0 * ((states >> v) & 1))
    for q, h in fields:
        energy += h * (1.0 - 2.0 * ((states >> q) & 1))
    i_min, i_max = int(energy.argmin()), int(energy.argmax())
    e_min, e_max = float(energy[i_min]), float(energy[i_max])
    n_ground = int(np.count_nonzero(np.abs(energy - e_min) < 1e-9))
    ground_bits = [(i_min >> q) & 1 for q in range(n)]  # entry q = bit of qubit q
    del states, energy

    problem = {
        "name": name,
        "n_qubits": n,
        "edges": edges,
        "fields": fields,
        "e_min": e_min,
        "e_max": e_max,
        "ground_state_bits": ground_bits,
        "max_state_bits": [(i_max >> q) & 1 for q in range(n)],
        "n_degenerate_ground_states": n_ground,
        "meta": {
            "rows": rows, "cols": cols, "inst": inst, "seed": seed, "periodic": False,
            "n_nn_edges": len(nn_pairs), "n_nnn_edges": len(nnn_pairs),
            "nn_std": NN_STD, "nnn_std": NNN_STD, "field_std": FIELD_STD,
        },
    }
    text = json.dumps(problem, indent=2)
    json_path = OUT_DIR / f"{name}.json"
    if json_path.exists() and json_path.read_text() != text:
        raise SystemExit(f"{json_path.name}: regenerated instance differs from the existing file")
    json_path.write_text(text)

    # ---- picture: NN solid, NNN dashed, edge color/width = J, node color = h
    pos = {q: (q % cols, -(q // cols)) for q in range(n)}
    j_max = max(abs(w) for _, _, w in edges)
    h_max = max(abs(h) for _, h in fields)
    fig, ax = plt.subplots(figsize=(1.6 * cols + 2.5, 1.6 * rows + 1.2))
    for k, (u, v, w) in enumerate(edges):
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ax.plot([x0, x1], [y0, y1], color=plt.cm.RdBu_r(0.5 + 0.5 * w / j_max),
                linewidth=0.8 + 4.0 * abs(w) / j_max, linestyle="-" if k < len(nn_pairs) else "--", zorder=1)
    sc = ax.scatter([pos[q][0] for q in range(n)], [pos[q][1] for q in range(n)], c=[h for _, h in fields],
                    cmap="PiYG", vmin=-h_max, vmax=h_max, s=420, edgecolors="black", zorder=2)
    for q in range(n):  # qubit index and its ground-state spin
        ax.text(pos[q][0], pos[q][1], f"{q}\n{'-' if ground_bits[q] else '+'}", ha="center", va="center",
                fontsize=8, zorder=3)
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-j_max, j_max)), ax=ax,
                 fraction=0.04, pad=0.02, label="J (solid NN, dashed NNN)")
    fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, label="h")
    ax.set_title(f"{name}\n{n} qubits, {len(nn_pairs)} NN + {len(nnn_pairs)} NNN edges, "
                 f"E_min={e_min:.4f}, E_max={e_max:.4f}  (node: index, ground-state spin)", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}: {len(edges)} edges, E_min={e_min:.6f} ({n_ground} ground states), E_max={e_max:.6f}")
