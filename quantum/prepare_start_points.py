# TC3 시작점 준비 (GPU): 한 케이스(configs/TC3.json의 grid·instance·const)에서 두 방식의 양자컴퓨터 학습 시작점을 만든다.
# standard 방식은 상수 초기값과 전체 회로, PADO 방식은 LWPP 사전학습 → 증류를 거친 θ와 증류된 회로.
# 두 시작점의 노이즈 없는 exact 에너지도 저장한다. 양자컴퓨터 학습은 train_qpu.ipynb.
# Usage: python quantum/prepare_start_points.py --config configs/TC3.json
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
try:
    from hipauli import CliffordGate, PauliRotation, PauliSum, compile_program
except ImportError:
    from padopauli import CliffordGate, PauliRotation, PauliSum, compile_program

from common import REPO, env_info, new_run_dir, save_json, save_qasm

cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
P_LAYERS, CONST, LR = cfg["p_layers"], cfg["const"], cfg["lr"]             # 초기값: RZZ·RZ 각도 = +c, RX 각도 = −c
LWPP_STEPS, MAX_WEIGHT = cfg["lwpp"]["steps"], cfg["lwpp"]["max_weight"]   # LWPP: weight > max_weight인 Pauli 항을 버리는 근사
LWPP_PRESET, LWPP_DTYPE = cfg["lwpp"]["preset"], cfg["lwpp"]["dtype"]
THRESHOLD = cfg["distill_threshold"]                                       # 증류: |θ| <= THRESHOLD인 게이트를 회로에서 제거
EXACT_DEVICE = cfg["exact_device"]
torch.set_default_dtype(torch.float64)  # PennyLane의 torch 인터페이스는 결과를 torch 기본 dtype으로 돌려준다

# ---- 문제 (전수 조사 해 E_min, E_max 포함)
inst = cfg["instance"]
problem_path = REPO / "problems" / f"{cfg['grid']}_inst{inst}_seed{cfg['seed_base'] + inst}.json"
prob = json.loads(problem_path.read_text())
n, e_min, e_max = prob["n_qubits"], prob["e_min"], prob["e_max"]
out = new_run_dir(cfg["name"], problem_path, "start_points")
tag = f"[{cfg['name']} {prob['name']} start_points]"

# ---- ma-QAOA 회로: 모든 큐비트에 H, 층마다 RZZ(결합) → RZ(필드) → RX(혼합), 게이트마다 독립 각도
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

# ---- 1/3 PADO LWPP 사전학습 (고전 GPU, 양자 자원 0)
print(f"{tag} 1/3 LWPP 사전학습: max_weight {MAX_WEIGHT}, {LWPP_STEPS} step, {LWPP_DTYPE}", flush=True)
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
n_terms = int(program.psum_union.x_mask.shape[0])  # compile 뒤 남은 Pauli 항 수 (zero-filter 이후)

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
print(f"{tag} compile {compile_s:.1f}s terms={n_terms}  {LWPP_STEPS} step {lwpp_train_s:.1f}s  E_lwpp {lwpp_energy[0]:+.4f} -> {lwpp_energy[-1]:+.4f}", flush=True)

# ---- 2/3 증류: |θ| <= THRESHOLD인 회전 게이트 제거, 남은 각도를 0..n_active-1로 다시 번호 매김
active = [kk for kk in range(n_params) if abs(theta_lwpp[kk]) > THRESHOLD]
new_idx = {old: new for new, old in enumerate(active)}
pruned = [g for g in gates if g["gate"] == "H"]
for g in gates[n:]:
    if g["param"] in new_idx:
        pruned.append({"gate": g["gate"], "wires": g["wires"], "param": new_idx[g["param"]]})
theta_pado_safe_start = theta_lwpp[active]
print(f"{tag} 2/3 증류: 남은 각도 {len(active)}/{n_params}  (step당 회로: standard {2 * n_params + 1}, PADO {2 * len(active) + 1})", flush=True)

# ---- 3/3 두 시작점의 노이즈 없는 exact 에너지 (PennyLane 상태벡터)
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


e_standard_start = float(energy(torch.tensor(theta_init), gates))
e_lwpp_final = float(energy(torch.tensor(theta_lwpp), gates))
e_pado_safe_start = float(energy(torch.tensor(theta_pado_safe_start), pruned))
ar = lambda e: (e_max - e) / (e_max - e_min)
print(f"{tag} 3/3 노이즈 없는 AR: standard 시작 {ar(e_standard_start):.4f} | LWPP 직후 {ar(e_lwpp_final):.4f} | PADO 시작(증류 후) {ar(e_pado_safe_start):.4f}", flush=True)

# ---- 저장
save_json(out / "circuit.json", {
    "n_qubits": n, "p_layers": P_LAYERS, "n_params": n_params,
    "convention": "R_P(theta) = exp(-i theta P / 2); RZZ = IsingZZ, initial state |0..0>; observable = sum J ZZ + sum h Z",
    "gates": gates, "pruned_gates": pruned, "edges": prob["edges"], "fields": prob["fields"],
})
save_json(out / "params.json", {
    "run": "start_points", "status": "ok", "experiment": cfg["name"], "problem": prob["name"], "e_min": e_min, "e_max": e_max,
    "config": {"p_layers": P_LAYERS, "const": CONST, "lwpp_steps": LWPP_STEPS, "max_weight": MAX_WEIGHT,
               "lwpp_preset": LWPP_PRESET, "lwpp_dtype": LWPP_DTYPE, "threshold": THRESHOLD, "lr": LR},
    "versions": env_info(),
    "lwpp": {"compile_s": compile_s, "n_terms": n_terms, "train_s": lwpp_train_s, "peak_gb": lwpp_peak_gb,
             "energy": lwpp_energy},
    "theta_init": theta_init.tolist(), "theta_lwpp_final": theta_lwpp.tolist(),
    "active_params": active, "n_active": len(active), "theta_pado_safe_start": theta_pado_safe_start.tolist(),
    "exact_energy": {"standard_start": e_standard_start, "lwpp_final": e_lwpp_final, "pado_safe_start": e_pado_safe_start},
    "circuits_per_step": {"standard": 2 * n_params + 1, "pado_safe": 2 * len(active) + 1},
})
save_qasm(out / "qasm" / "standard_start.qasm", n, gates, theta_init)
save_qasm(out / "qasm" / "pado_safe_start.qasm", n, pruned, theta_pado_safe_start)
print(f"{tag} 끝: 저장 {out.relative_to(REPO)}", flush=True)
