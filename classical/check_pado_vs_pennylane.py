# Convention check: PADO without truncation (max_weight = n) must equal the PennyLane exact energy and gradient.
# Same ma-QAOA circuit and observable as pado_safe.py / standard.py, on the 3x3 instance (seconds).
# The check runs on the host engine (preset "cpu", float64). The NPU (preset "tenstorrent") is float32 only:
# its difference on the same circuit is printed, not judged.
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
from padopauli import CliffordGate, PauliRotation, PauliSum, compile_program

PROBLEM = Path(__file__).resolve().parents[1] / "problems" / "grid3x3_inst0_seed120490.json"
P_LAYERS = 3
EXACT_DEVICE = "lightning.qubit"
N_THETAS = 4
torch.set_default_dtype(torch.float64)  # PennyLane's torch interface returns results in the torch default dtype

# ---- problem
prob = json.loads(PROBLEM.read_text())
n = prob["n_qubits"]

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

# ---- PADO, full weight (exact Pauli propagation): host float64, NPU float32
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
program = compile_program(circuit=pado_circuit, observables=[obs], preset="cpu",
                          preset_overrides={"max_weight": n, "dtype": "float64"}, verbosity=0)
program_npu = compile_program(circuit=pado_circuit, observables=[obs], preset="tenstorrent",
                              preset_overrides={"max_weight": n, "dtype": "float32"}, verbosity=0)

# ---- PennyLane exact, adjoint differentiation
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


# ---- compare on a few random parameter vectors
rng = np.random.default_rng(0)
worst_e, worst_g, worst_e_npu, worst_g_npu = 0.0, 0.0, 0.0, 0.0
for i in range(N_THETAS):
    theta_np = rng.uniform(-0.5, 0.5, size=n_params)

    th_pado = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    e_pado = program.expvals(th_pado, diff_mode="vjp")[0]
    e_pado.backward()

    th_npu = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    e_npu = program_npu.expvals(th_npu, diff_mode="vjp")[0]
    e_npu.backward()

    th_pl = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    e_pl = energy(th_pl)
    e_pl.backward()

    e_pado, e_npu, e_pl = e_pado.detach(), e_npu.detach(), e_pl.detach()
    de = abs(float(e_pado) - float(e_pl))
    dg = float((th_pado.grad - th_pl.grad).abs().max())
    de_npu = abs(float(e_npu) - float(e_pl))
    dg_npu = float((th_npu.grad - th_pl.grad).abs().max())
    worst_e, worst_g = max(worst_e, de), max(worst_g, dg)
    worst_e_npu, worst_g_npu = max(worst_e_npu, de_npu), max(worst_g_npu, dg_npu)
    print(f"theta {i}: E_pado={float(e_pado):+.10f} E_pl={float(e_pl):+.10f} |dE|={de:.2e} max|dgrad|={dg:.2e}"
          f"  | NPU E={float(e_npu):+.10f} |dE|={de_npu:.2e} max|dgrad|={dg_npu:.2e}")

print(f"NPU float32 (not judged): worst |dE|={worst_e_npu:.2e}, worst |dgrad|={worst_g_npu:.2e}")
print(f"{n_params} params, host float64: worst |dE|={worst_e:.2e}, worst |dgrad|={worst_g:.2e}")
if worst_e > 1e-8 or worst_g > 1e-8:
    sys.exit("FAIL: PADO and PennyLane disagree (gate or observable convention mismatch)")
print("PASS")
