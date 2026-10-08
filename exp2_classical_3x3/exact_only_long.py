# Experiment 2 (classical 3x3), exact-only run extended until it reaches the SAFE arm's final AR (or MAX_STEPS).
# Reference for the QPU run of experiment 3: how many exact steps (QPU gradient evaluations) SAFE saves.
# Same circuit, init and optimizer as exact_only.py; the first 20 steps therefore repeat that run.
# Usage: python exp2_classical_3x3/exact_only_long.py <inst 0-4>      (needs <problem>/safe/params.json first)
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from common import REPO, device_tag, new_run_dir, save_json, save_qasm

P_LAYERS = 3
CONSTS = [0.01, 0.05, 0.1, 0.2, 0.4]  # gamma = +c, beta = -c (paper Sec. III-B)
MAX_STEPS = 2000        # stop here if the target AR is never reached
LR = 0.02  # Adam, default betas (paper footnote 1)
EXACT_DEVICE = "lightning.amdgpu"
torch.set_default_dtype(torch.float64)  # PennyLane's torch interface returns results in the torch default dtype

problem_path = REPO / "problems" / f"grid3x3_inst{int(sys.argv[1])}_seed{120490 + int(sys.argv[1])}.json"
experiment = "exp2_classical_3x3"

# ---- problem and per-init target: the AR the SAFE arm reached after its exact fine-tuning
prob = json.loads(problem_path.read_text())
n, e_min, e_max = prob["n_qubits"], prob["e_min"], prob["e_max"]
safe = json.loads((REPO / f"results_on_{device_tag()}" / experiment / problem_path.stem / "safe" / "params.json").read_text())
target = {run["c"]: run["ar"][-1] for run in safe["inits"]}
out = new_run_dir(experiment, problem_path, "exact_only_long")

# ---- ma-QAOA circuit: H on all, then per layer one angle per ZZ edge, per Z field, per X mixer
gates = [{"gate": "H", "wires": [q]} for q in range(n)]
k = 0
for layer in range(P_LAYERS):
    for u, v, w in prob["edges"]:
        gates.append({"gate": "RZZ", "wires": [u, v], "param": k}); k += 1
    for q, h in prob["fields"]:
        gates.append({"gate": "RZ", "wires": [q], "param": k}); k += 1
    for q in range(n):
        gates.append({"gate": "RX", "wires": [q], "param": k}); k += 1
n_params = k
save_json(out / "circuit.json", {
    "n_qubits": n, "p_layers": P_LAYERS, "n_params": n_params,
    "convention": "R_P(theta) = exp(-i theta P / 2); RZZ = IsingZZ, initial state |0..0>; observable = sum J ZZ + sum h Z",
    "gates": gates, "edges": prob["edges"], "fields": prob["fields"],
})

# ---- exact energy: PennyLane state vector on the AMD GPU, adjoint gradient
H = qml.Hamiltonian([w for _, _, w in prob["edges"]] + [h for _, h in prob["fields"]],
                    [qml.Z(u) @ qml.Z(v) for u, v, _ in prob["edges"]] + [qml.Z(q) for q, _ in prob["fields"]])
dev = qml.device(EXACT_DEVICE, wires=n)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def energy(theta):
    for g in gates:
        if g["gate"] == "H":
            qml.Hadamard(g["wires"][0])
        elif g["gate"] == "RZZ":
            qml.IsingZZ(theta[g["param"]], wires=g["wires"])
        elif g["gate"] == "RZ":
            qml.RZ(theta[g["param"]], wires=g["wires"][0])
        elif g["gate"] == "RX":
            qml.RX(theta[g["param"]], wires=g["wires"][0])
    return qml.expval(H)


# ---- training: one init at a time; step t records the energy before update t; stop at the first t with AR >= target
runs = []
for c in CONSTS:
    theta_init = np.zeros(n_params)
    for g in gates[n:]:
        theta_init[g["param"]] = -c if g["gate"] == "RX" else c
    theta = torch.tensor(theta_init, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    energies, ar, hit = [], [], None
    t0 = time.perf_counter()
    for t in range(MAX_STEPS + 1):
        opt.zero_grad()
        e = energy(theta)
        energies.append(float(e.detach()))
        ar.append((e_max - energies[-1]) / (e_max - e_min))
        if ar[-1] >= target[c]:
            hit = t
            break
        if t < MAX_STEPS:
            e.backward()
            opt.step()
        if t % 100 == 0:
            print(f"[exact_only_long c={c}] step {t:4d}  AR={ar[-1]:.4f}  target={target[c]:.4f}", flush=True)
    wall_s = time.perf_counter() - t0
    print(f"[exact_only_long c={c}] " + (f"reached target at step {hit}" if hit is not None else f"not reached in {MAX_STEPS} steps")
          + f"  ({wall_s:.1f}s)", flush=True)
    runs.append({
        "label": f"const_{c}", "c": c, "n_active": n_params, "target_ar": target[c], "steps_to_target": hit,
        "theta_init": theta_init.tolist(), "theta_final": theta.detach().tolist(),
        "energy": energies, "ar": ar, "wall_s": wall_s,
    })
    save_qasm(out / "qasm" / f"exact_only_long_c{c}_init.qasm", n, gates, theta_init)
    save_qasm(out / "qasm" / f"exact_only_long_c{c}_final.qasm", n, gates, theta.detach().tolist())

# ---- save
save_json(out / "params.json", {
    "arm": "exact_only_long", "problem": prob["name"], "e_min": e_min, "e_max": e_max,
    "config": {"p_layers": P_LAYERS, "consts": CONSTS, "max_steps": MAX_STEPS, "lr": LR, "exact_device": EXACT_DEVICE,
               "diff_method": "adjoint", "circuits_per_step": 2 * n_params + 1,
               "target": "per init: final AR of the safe arm (safe/params.json inits[i].ar[-1])"},
    "versions": {"torch": torch.__version__, "pennylane": qml.__version__},
    "inits": runs,
})
print(f"saved {out}")
