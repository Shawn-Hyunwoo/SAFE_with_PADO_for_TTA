# Experiment 1, arm A (exact-only): constant initialisation -> PennyLane exact training, EXACT_STEPS updates per init.
# Usage: python exp1_classical_5x5/exact_only.py <inst 0-4>
#        python exp1_classical_5x5/exact_only.py <problem.json> <exact_steps>     (smoke: results go to exp1_smoke)
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from common import new_run_dir, save_json, save_qasm

P_LAYERS = 3
CONSTS = [0.01, 0.05, 0.1, 0.2, 0.4]  # gamma = +c, beta = -c (paper Sec. III-B)
EXACT_STEPS = 20
LR = 0.02  # Adam, default betas (paper footnote 1)
EXACT_DEVICE = "lightning.amdgpu"
torch.set_default_dtype(torch.float64)  # PennyLane's torch interface returns results in the torch default dtype

REPO = Path(__file__).resolve().parents[1]
if len(sys.argv) == 2:
    problem_path = REPO / "problems" / f"grid5x5_inst{int(sys.argv[1])}_seed{120490 + int(sys.argv[1])}.json"
    experiment = "exp1_classical_5x5"
else:
    problem_path, EXACT_STEPS, experiment = Path(sys.argv[1]).resolve(), int(sys.argv[2]), "exp1_smoke"

# ---- problem
prob = json.loads(problem_path.read_text())
n, e_min, e_max = prob["n_qubits"], prob["e_min"], prob["e_max"]
out = new_run_dir(experiment, problem_path, "exact_only")

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


# ---- training: one init at a time; step t records the energy before update t (t = EXACT_STEPS is the final point)
runs = []
for c in CONSTS:
    theta_init = np.zeros(n_params)
    for g in gates[n:]:
        theta_init[g["param"]] = -c if g["gate"] == "RX" else c
    theta = torch.tensor(theta_init, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    energies, traj, step_s = [], [], []
    for t in range(EXACT_STEPS + 1):
        t0 = time.perf_counter()
        opt.zero_grad()
        e = energy(theta)
        energies.append(float(e.detach()))
        traj.append(theta.detach().tolist())
        if t < EXACT_STEPS:
            e.backward()
            opt.step()
        step_s.append(time.perf_counter() - t0)
        print(f"[exact_only c={c}] step {t:3d}  E={energies[-1]:+.6f}  AR={(e_max - energies[-1]) / (e_max - e_min):.4f}"
              f"  {step_s[-1]:.2f}s", flush=True)
    ar = [(e_max - x) / (e_max - e_min) for x in energies]
    runs.append({
        "label": f"const_{c}", "c": c, "n_active": n_params, "active_params": list(range(n_params)),
        "theta_init": theta_init.tolist(), "theta_final": traj[-1],
        "energy": energies, "ar": ar, "theta_traj": traj, "step_wall_s": step_s,
    })
    save_qasm(out / "qasm" / f"exact_only_c{c}_init.qasm", n, gates, theta_init)
    save_qasm(out / "qasm" / f"exact_only_c{c}_final.qasm", n, gates, traj[-1])

# ---- save
save_json(out / "params.json", {
    "arm": "exact_only", "problem": prob["name"], "e_min": e_min, "e_max": e_max,
    "config": {"p_layers": P_LAYERS, "consts": CONSTS, "exact_steps": EXACT_STEPS, "lr": LR,
               "exact_device": EXACT_DEVICE, "diff_method": "adjoint", "circuits_per_step": 2 * n_params + 1},
    "versions": {"torch": torch.__version__, "pennylane": qml.__version__},
    "inits": runs,
})
print(f"saved {out}")
