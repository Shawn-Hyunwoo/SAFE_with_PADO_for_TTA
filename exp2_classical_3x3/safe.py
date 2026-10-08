# Experiment 2 (classical 3x3), arm B (SAFE): same constant inits -> PADO LWPP pre-training -> distillation ->
# PennyLane exact fine-tuning. Same code as exp1_classical_5x5/safe.py except size, MAX_WEIGHT, LWPP_STEPS.
# Usage: python exp2_classical_3x3/safe.py <inst 0-4>
import json
import sys
import time
from pathlib import Path

import numpy as np
import padopauli
import pennylane as qml
import torch
from padopauli import CliffordGate, PauliRotation, PauliSum, compile_program

from common import new_run_dir, save_json, save_qasm

P_LAYERS = 3
CONSTS = [0.01, 0.05, 0.1, 0.2, 0.4]  # gamma = +c, beta = -c (paper Sec. III-B)
LWPP_STEPS = 100
MAX_WEIGHT = 3          # LWPP truncation: Pauli words of weight > MAX_WEIGHT are dropped
LWPP_PRESET = "gpu"
LWPP_DTYPE = "float32"
THRESHOLD = 0.3         # distillation: angles with |theta| <= THRESHOLD are removed from the circuit
EXACT_STEPS = 20
LR = 0.02               # Adam, default betas, all stages (paper footnote 1)
EXACT_DEVICE = "lightning.amdgpu"
torch.set_default_dtype(torch.float64)  # PennyLane's torch interface returns results in the torch default dtype

REPO = Path(__file__).resolve().parents[1]
problem_path = REPO / "problems" / f"grid3x3_inst{int(sys.argv[1])}_seed{120490 + int(sys.argv[1])}.json"
experiment = "exp2_classical_3x3"

# ---- problem
prob = json.loads(problem_path.read_text())
n, e_min, e_max = prob["n_qubits"], prob["e_min"], prob["e_max"]
out = new_run_dir(experiment, problem_path, "safe")

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

theta_init = np.zeros((len(CONSTS), n_params))
for i, c in enumerate(CONSTS):
    for g in gates[n:]:
        theta_init[i, g["param"]] = -c if g["gate"] == "RX" else c

# ---- stage 1: PADO LWPP surrogate (compile once, then train all inits as one batch)
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
compile_peak_gb = torch.cuda.max_memory_allocated() / 1e9
n_terms = int(program.psum_union.x_mask.shape[0])  # Pauli terms kept after compile (post zero-filter)
print(f"[lwpp] compile {compile_s:.1f}s  terms={n_terms}  peak={compile_peak_gb:.2f} GB", flush=True)

thetas = torch.tensor(theta_init, device="cuda", requires_grad=True)
opt = torch.optim.Adam([thetas], lr=LR)
lwpp_energy = []
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for step in range(LWPP_STEPS):
    e = program.expvals(thetas, diff_mode="vjp")[:, 0]
    opt.zero_grad()
    e.sum().backward()  # sum: each init gets its own gradient
    opt.step()
    lwpp_energy.append(e.detach().cpu().tolist())
    if step % 20 == 0 or step == LWPP_STEPS - 1:
        print(f"[lwpp] step {step:4d}  E_lwpp={[round(x, 4) for x in lwpp_energy[-1]]}", flush=True)
torch.cuda.synchronize()
lwpp_train_s = time.perf_counter() - t0
lwpp_train_peak_gb = torch.cuda.max_memory_allocated() / 1e9
theta_lwpp = thetas.detach().cpu().numpy()
lwpp_energy_final = program.expvals(torch.tensor(theta_lwpp, device="cuda"))[:, 0].cpu().tolist()

# ---- exact energy on any (possibly distilled) gate list: PennyLane state vector on the AMD GPU, adjoint gradient
H = qml.Hamiltonian([w for _, _, w in prob["edges"]] + [h for _, h in prob["fields"]],
                    [qml.Z(u) @ qml.Z(v) for u, v, _ in prob["edges"]] + [qml.Z(q) for q, _ in prob["fields"]])
dev = qml.device(EXACT_DEVICE, wires=n)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
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


runs = []
for i, c in enumerate(CONSTS):
    # exact energy at the LWPP-final parameters before distillation (LWPP-only reference)
    exact_lwpp_only = float(energy(torch.tensor(theta_lwpp[i]), gates))

    # ---- stage 2: distillation -- drop rotations with |theta| <= THRESHOLD, renumber the kept ones 0..n_active-1
    active = [kk for kk in range(n_params) if abs(theta_lwpp[i, kk]) > THRESHOLD]
    new_idx = {old: new for new, old in enumerate(active)}
    pruned = [g for g in gates if g["gate"] == "H"]
    for g in gates[n:]:
        if g["param"] in new_idx:
            pruned.append({"gate": g["gate"], "wires": g["wires"], "param": new_idx[g["param"]]})
    print(f"[distill c={c}] kept {len(active)}/{n_params} angles" + ("  (all removed: energy stays constant)" if not active else ""), flush=True)

    # ---- stage 3: exact fine-tuning on the pruned circuit; step t records the energy before update t
    theta = torch.tensor(theta_lwpp[i, active], requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    energies, traj, step_s = [], [], []
    for t in range(EXACT_STEPS + 1):
        t0 = time.perf_counter()
        opt.zero_grad()
        e = energy(theta, pruned)
        energies.append(float(e.detach()))
        traj.append(theta.detach().tolist())
        if t < EXACT_STEPS and active:  # nothing to train when distillation removed every angle
            e.backward()
            opt.step()
        step_s.append(time.perf_counter() - t0)
        print(f"[safe c={c}] step {t:3d}  E={energies[-1]:+.6f}  AR={(e_max - energies[-1]) / (e_max - e_min):.4f}"
              f"  {step_s[-1]:.2f}s", flush=True)

    theta_final_full = np.zeros(n_params)  # removed angles stay 0 (identity), so the full circuit gives the same state
    theta_final_full[active] = traj[-1]
    runs.append({
        "label": f"const_{c}", "c": c, "n_active": len(active), "active_params": active,
        "theta_init": theta_init[i].tolist(), "theta_lwpp_final": theta_lwpp[i].tolist(),
        "lwpp_energy": [row[i] for row in lwpp_energy], "lwpp_energy_final": lwpp_energy_final[i],
        "exact_energy_lwpp_final": exact_lwpp_only,
        "theta_final": theta_final_full.tolist(),
        "energy": energies, "ar": [(e_max - x) / (e_max - e_min) for x in energies],
        "theta_traj_active": traj, "step_wall_s": step_s,
    })
    save_qasm(out / "qasm" / f"safe_c{c}_init.qasm", n, gates, theta_init[i])
    save_qasm(out / "qasm" / f"safe_c{c}_exact_start.qasm", n, pruned, traj[0])
    save_qasm(out / "qasm" / f"safe_c{c}_final.qasm", n, pruned, traj[-1])

# ---- save
save_json(out / "circuit.json", {
    "n_qubits": n, "p_layers": P_LAYERS, "n_params": n_params,
    "convention": "R_P(theta) = exp(-i theta P / 2); RZZ = IsingZZ, initial state |0..0>; observable = sum J ZZ + sum h Z",
    "gates": gates, "edges": prob["edges"], "fields": prob["fields"],
    "distilled": "per init: keep gates whose param is in params.json inits[i].active_params, renumbered 0..n_active-1 in order",
})
save_json(out / "params.json", {
    "arm": "safe", "problem": prob["name"], "e_min": e_min, "e_max": e_max,
    "config": {"p_layers": P_LAYERS, "consts": CONSTS, "lwpp_steps": LWPP_STEPS, "max_weight": MAX_WEIGHT,
               "lwpp_preset": LWPP_PRESET, "lwpp_dtype": LWPP_DTYPE, "threshold": THRESHOLD,
               "exact_steps": EXACT_STEPS, "lr": LR, "exact_device": EXACT_DEVICE, "diff_method": "adjoint"},
    "versions": {"torch": torch.__version__, "pennylane": qml.__version__, "padopauli": padopauli.__version__},
    "lwpp": {"compile_s": compile_s, "compile_peak_gb": compile_peak_gb, "n_terms": n_terms,
             "train_s": lwpp_train_s, "train_peak_gb": lwpp_train_peak_gb},
    "inits": runs,
})
print(f"saved {out}")
