# Experiment 3 (3x3 QPU run, one case: instance 0, c = 0.1): start points of the two QPU arms, computed on the GPU.
# exact-only starts from theta_init on the full circuit; SAFE starts from the PADO LWPP pre-trained and distilled theta
# on the pruned circuit. Also saves both start points' noiseless exact energies. The QPU training is train_qpu.py.
# Usage: python exp3_quantum_3x3/prepare_start_points.py
import json
import time
from pathlib import Path

import numpy as np
import padopauli
import pennylane as qml
import torch
from padopauli import CliffordGate, PauliRotation, PauliSum, compile_program

from common import new_run_dir, save_json, save_qasm

P_LAYERS = 3
CONST = 0.1             # gamma = +c, beta = -c; a single init for the one-shot QPU runs
LWPP_STEPS = 100
MAX_WEIGHT = 3          # LWPP truncation: Pauli words of weight > MAX_WEIGHT are dropped
LWPP_PRESET = "gpu"
LWPP_DTYPE = "float32"
THRESHOLD = 0.3         # distillation: angles with |theta| <= THRESHOLD are removed from the circuit
LR = 0.02               # Adam, default betas (paper footnote 1)
EXACT_DEVICE = "lightning.amdgpu"
torch.set_default_dtype(torch.float64)  # PennyLane's torch interface returns results in the torch default dtype

REPO = Path(__file__).resolve().parents[1]
problem_path = REPO / "problems" / "grid3x3_inst0_seed120490.json"

# ---- problem
prob = json.loads(problem_path.read_text())
n, e_min, e_max = prob["n_qubits"], prob["e_min"], prob["e_max"]
out = new_run_dir("exp3_quantum_3x3", problem_path, "start_points")

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

theta_init = np.zeros(n_params)
for g in gates[n:]:
    theta_init[g["param"]] = -CONST if g["gate"] == "RX" else CONST

# ---- PADO LWPP pre-training
pado_circuit = []
for g in gates:
    if g["gate"] == "H":
        pado_circuit.append(CliffordGate("H", g["wires"]))
    else:
        pado_circuit.append(PauliRotation({"RZZ": "ZZ", "RZ": "Z", "RX": "X"}[g["gate"]], g["wires"], param_idx=g["param"]))
obs = PauliSum(n_qubits=n)
for u, v, w in prob["edges"]:
    obs.add_from_str("ZZ", w, qubits=[u, v])
for q, h in prob["fields"]:
    obs.add_from_str("Z", h, qubits=[q])

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
program = compile_program(circuit=pado_circuit, observables=[obs], preset=LWPP_PRESET,
                          preset_overrides={"max_weight": MAX_WEIGHT, "dtype": LWPP_DTYPE})
compile_s = time.perf_counter() - t0
n_terms = int(program.psum_union.x_mask.shape[0])  # Pauli terms kept after compile (post zero-filter)

theta = torch.tensor(theta_init, device="cuda", requires_grad=True)
opt = torch.optim.Adam([theta], lr=LR)
lwpp_energy = []
t0 = time.perf_counter()
for step in range(LWPP_STEPS):
    e = program.expvals(theta, diff_mode="vjp")[0]
    opt.zero_grad()
    e.backward()
    opt.step()
    lwpp_energy.append(float(e.detach()))
torch.cuda.synchronize()
lwpp_train_s = time.perf_counter() - t0
lwpp_peak_gb = torch.cuda.max_memory_allocated() / 1e9
theta_lwpp = theta.detach().cpu().numpy()
print(f"[lwpp] compile {compile_s:.1f}s terms={n_terms}  {LWPP_STEPS} steps {lwpp_train_s:.1f}s  E_lwpp {lwpp_energy[0]:+.4f} -> {lwpp_energy[-1]:+.4f}")

# ---- distillation: drop rotations with |theta| <= THRESHOLD, renumber the kept ones 0..n_active-1
active = [kk for kk in range(n_params) if abs(theta_lwpp[kk]) > THRESHOLD]
new_idx = {old: new for new, old in enumerate(active)}
pruned = [g for g in gates if g["gate"] == "H"]
for g in gates[n:]:
    if g["param"] in new_idx:
        pruned.append({"gate": g["gate"], "wires": g["wires"], "param": new_idx[g["param"]]})
theta_safe_start = theta_lwpp[active]
print(f"[distill] kept {len(active)}/{n_params} angles")

# ---- noiseless exact energies of both QPU starting points (PennyLane state vector)
H = qml.Hamiltonian([w for _, _, w in prob["edges"]] + [h for _, h in prob["fields"]],
                    [qml.Z(u) @ qml.Z(v) for u, v, _ in prob["edges"]] + [qml.Z(q) for q, _ in prob["fields"]])
dev = qml.device(EXACT_DEVICE, wires=n)


@qml.qnode(dev, interface="torch")
def energy(theta, circuit_gates):
    for g in circuit_gates:
        if g["gate"] == "H":
            qml.Hadamard(g["wires"][0])
        elif g["gate"] == "RZZ":
            qml.IsingZZ(theta[g["param"]], wires=g["wires"])
        elif g["gate"] == "RZ":
            qml.RZ(theta[g["param"]], wires=g["wires"][0])
        elif g["gate"] == "RX":
            qml.RX(theta[g["param"]], wires=g["wires"][0])
    return qml.expval(H)


e_exact_only_start = float(energy(torch.tensor(theta_init), gates))
e_lwpp_final = float(energy(torch.tensor(theta_lwpp), gates))
e_safe_start = float(energy(torch.tensor(theta_safe_start), pruned))
ar = lambda e: (e_max - e) / (e_max - e_min)
print(f"[exact] AR exact-only start {ar(e_exact_only_start):.4f} | LWPP final {ar(e_lwpp_final):.4f} | SAFE start {ar(e_safe_start):.4f}")

# ---- save
save_json(out / "circuit.json", {
    "n_qubits": n, "p_layers": P_LAYERS, "n_params": n_params,
    "convention": "R_P(theta) = exp(-i theta P / 2); RZZ = IsingZZ, initial state |0..0>; observable = sum J ZZ + sum h Z",
    "gates": gates, "pruned_gates": pruned, "edges": prob["edges"], "fields": prob["fields"],
})
save_json(out / "params.json", {
    "problem": prob["name"], "e_min": e_min, "e_max": e_max,
    "config": {"p_layers": P_LAYERS, "const": CONST, "lwpp_steps": LWPP_STEPS, "max_weight": MAX_WEIGHT,
               "lwpp_preset": LWPP_PRESET, "lwpp_dtype": LWPP_DTYPE, "threshold": THRESHOLD, "lr": LR},
    "versions": {"torch": torch.__version__, "pennylane": qml.__version__, "padopauli": padopauli.__version__},
    "lwpp": {"compile_s": compile_s, "n_terms": n_terms, "train_s": lwpp_train_s, "peak_gb": lwpp_peak_gb,
             "energy": lwpp_energy},
    "theta_init": theta_init.tolist(), "theta_lwpp_final": theta_lwpp.tolist(),
    "active_params": active, "n_active": len(active), "theta_safe_start": theta_safe_start.tolist(),
    "exact_energy": {"exact_only_start": e_exact_only_start, "lwpp_final": e_lwpp_final, "safe_start": e_safe_start},
    "circuits_per_step": {"exact_only": 2 * n_params + 1, "safe": 2 * len(active) + 1},
})
save_qasm(out / "qasm" / "exact_only_start.qasm", n, gates, theta_init)
save_qasm(out / "qasm" / "safe_start.qasm", n, pruned, theta_safe_start)
print(f"saved {out}")
