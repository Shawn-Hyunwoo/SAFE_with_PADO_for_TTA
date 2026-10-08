# Experiment 1 report: per (instance, init) AR at the first/last exact step, first-hit steps, circuit-count cost.
# Reads results_on_<TAG>/<experiment>/*/{exact_only,safe}/params.json, writes <experiment>/report/.
# Usage: python exp1_classical_5x5/report.py [experiment]      (default exp1_classical_5x5; exp1_smoke works too)
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import REPO, device_tag

experiment = sys.argv[1] if len(sys.argv) > 1 else "exp1_classical_5x5"
root = REPO / f"results_on_{device_tag()}" / experiment
problems = sorted(d for d in root.iterdir() if (d / "exact_only" / "params.json").exists() and (d / "safe" / "params.json").exists())
(root / "report").mkdir(exist_ok=True)

# ---- metrics
rows = []
for d in problems:
    A = json.loads((d / "exact_only" / "params.json").read_text())
    B = json.loads((d / "safe" / "params.json").read_text())
    for ra, rb in zip(A["inits"], B["inits"]):
        ar_a, ar_b = np.array(ra["ar"]), np.array(rb["ar"])
        target = ar_a[-1]  # AR the exact-only arm reaches after its full budget
        hit_a, hit_b = np.flatnonzero(ar_a >= target), np.flatnonzero(ar_b >= target)
        tau_a = int(hit_a[0])
        tau_b = int(hit_b[0]) if hit_b.size else None  # None = SAFE never reached the target (failure)
        cps_a, cps_b = 2 * ra["n_active"] + 1, 2 * rb["n_active"] + 1  # parameter-shift circuits per step
        cost_a, cost_b = tau_a * cps_a, (tau_b * cps_b if tau_b is not None else None)
        rows.append({
            "problem": A["problem"], "c": ra["c"], "n_params": ra["n_active"], "n_active_safe": rb["n_active"],
            "ar_exact_only_first": ar_a[0], "ar_exact_only_last": ar_a[-1],
            "ar_safe_lwpp_only": (B["e_max"] - rb["exact_energy_lwpp_final"]) / (B["e_max"] - B["e_min"]),
            "ar_safe_first": ar_b[0], "ar_safe_last": ar_b[-1],
            "tau_target_exact_only": tau_a, "tau_target_safe": tau_b,
            "tau099_exact_only": int(np.flatnonzero(ar_a >= 0.99 * ar_a.max())[0]),
            "tau099_safe": int(np.flatnonzero(ar_b >= 0.99 * ar_b.max())[0]),
            "circuits_per_step_exact_only": cps_a, "circuits_per_step_safe": cps_b,
            "circuits_to_target_exact_only": cost_a, "circuits_to_target_safe": cost_b,
            "circuit_saving": (1 - cost_b / cost_a) if (cost_b is not None and cost_a > 0) else None,
        })

with open(root / "report" / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

n_fail = sum(r["tau_target_safe"] is None for r in rows)
print(f"{len(rows)} (instance, init) pairs, SAFE failed to reach the exact-only final AR in {n_fail}")
for key in ["ar_exact_only_first", "ar_exact_only_last", "ar_safe_lwpp_only", "ar_safe_first", "ar_safe_last", "n_active_safe"]:
    print(f"  mean {key:22s} = {np.mean([r[key] for r in rows]):.4f}")
savings = [r["circuit_saving"] for r in rows if r["circuit_saving"] is not None]
if savings:
    print(f"  mean circuit_saving        = {np.mean(savings):.4f}  (over {len(savings)} pairs)")

# ---- figure: exact energy per step, both arms, brute-force E_min as the reference line
fig, axes = plt.subplots(1, len(problems), figsize=(4.2 * len(problems), 3.6), squeeze=False)
for ax, d in zip(axes[0], problems):
    A = json.loads((d / "exact_only" / "params.json").read_text())
    B = json.loads((d / "safe" / "params.json").read_text())
    for i, (ra, rb) in enumerate(zip(A["inits"], B["inits"])):
        ax.plot(ra["energy"], color="black", alpha=0.6, linewidth=1, label="exact-only" if i == 0 else None)
        ax.plot(rb["energy"], color="tab:orange", alpha=0.8, linewidth=1, label="SAFE" if i == 0 else None)
    ax.axhline(A["e_min"], color="tab:blue", linestyle="--", linewidth=1, label="brute-force E_min")
    ax.set_title(A["problem"], fontsize=9)
    ax.set_xlabel("exact step")
axes[0][0].set_ylabel("exact energy")
axes[0][0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(root / "report" / "energy_curves.png", dpi=150)
print(f"saved {root / 'report'}")
