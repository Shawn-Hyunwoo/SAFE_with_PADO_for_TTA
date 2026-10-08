# Experiment 2 (classical 3x3) report: per (instance, init) AR at the first/last exact step, first-hit steps,
# circuit-count cost (same as exp1_classical_5x5/report.py), plus the extended exact-only run when present.
# Reads results_on_<TAG>/exp2_classical_3x3/*/{exact_only,safe,exact_only_long}/params.json, writes exp2_classical_3x3/report/.
# Usage: python exp2_classical_3x3/report.py
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import REPO, device_tag

experiment = "exp2_classical_3x3"
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

# ---- extended exact-only (exact_only_long.py): exact steps / circuits each arm needs to reach the SAFE final AR
long_problems = [d for d in problems if (d / "exact_only_long" / "params.json").exists()]
long_rows = []
for d in long_problems:
    A = json.loads((d / "exact_only" / "params.json").read_text())
    B = json.loads((d / "safe" / "params.json").read_text())
    L = json.loads((d / "exact_only_long" / "params.json").read_text())
    for ra, rb, rl in zip(A["inits"], B["inits"], L["inits"]):
        ar_b, ar_l = np.array(rb["ar"]), np.array(rl["ar"])
        steps_safe = int(np.flatnonzero(ar_b >= rl["target_ar"])[0])
        steps_long = rl["steps_to_target"]  # None = not reached within max_steps
        hit_first = np.flatnonzero(ar_l >= ar_b[0])
        cps_a, cps_b = 2 * rl["n_active"] + 1, 2 * rb["n_active"] + 1
        k = min(len(ra["energy"]), len(rl["energy"]))
        long_rows.append({
            "problem": A["problem"], "c": rl["c"], "target_ar_safe_last": rl["target_ar"],
            "steps_safe_to_target": steps_safe, "steps_exact_only_to_target": steps_long,
            "steps_exact_only_to_safe_first": int(hit_first[0]) if hit_first.size else None,
            "circuits_safe_to_target": steps_safe * cps_b,
            "circuits_exact_only_to_target": steps_long * cps_a if steps_long is not None else None,
            "circuit_saving_to_target": 1 - steps_safe * cps_b / (steps_long * cps_a) if steps_long else None,
            "max_dE_first_steps_vs_exact_only": float(np.abs(np.array(ra["energy"][:k]) - np.array(rl["energy"][:k])).max()),
        })

if long_rows:
    with open(root / "report" / "summary_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)
    reached = [r for r in long_rows if r["steps_exact_only_to_target"] is not None]
    print(f"extended exact-only: {len(reached)}/{len(long_rows)} pairs reached the SAFE final AR")
    for key in ["steps_safe_to_target", "steps_exact_only_to_target", "circuits_safe_to_target",
                "circuits_exact_only_to_target", "circuit_saving_to_target"]:
        print(f"  mean {key:32s} = {np.mean([r[key] for r in reached]):.4f}  (over reached pairs)")
    print(f"  max |dE| of the repeated first steps vs exact_only.py = {max(r['max_dE_first_steps_vs_exact_only'] for r in long_rows):.2e}")

    fig, axes = plt.subplots(1, len(long_problems), figsize=(4.2 * len(long_problems), 3.6), squeeze=False)
    for ax, d in zip(axes[0], long_problems):
        B = json.loads((d / "safe" / "params.json").read_text())
        L = json.loads((d / "exact_only_long" / "params.json").read_text())
        for i, (rb, rl) in enumerate(zip(B["inits"], L["inits"])):
            ax.plot(rl["energy"], color="black", alpha=0.6, linewidth=1, label="exact-only (until SAFE final AR)" if i == 0 else None)
            ax.plot(rb["energy"], color="tab:orange", alpha=0.8, linewidth=1, label="SAFE (20 steps)" if i == 0 else None)
        ax.axhline(L["e_min"], color="tab:blue", linestyle="--", linewidth=1, label="brute-force E_min")
        ax.set_title(L["problem"], fontsize=9)
        ax.set_xlabel("exact step")
    axes[0][0].set_ylabel("exact energy")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "report" / "energy_curves_long.png", dpi=150)
print(f"saved {root / 'report'}")
