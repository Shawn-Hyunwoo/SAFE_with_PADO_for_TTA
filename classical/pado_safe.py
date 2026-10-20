# PADO 방식: 같은 상수 초기값 → PADO-Pauli LWPP 대리 모델로 사전학습(고전 GPU) → 파라미터 증류 → exact(상태벡터) 미세조정.
# 양자 자원(환산 회로 수)은 마지막 단계에서만 든다. 초기값 c마다 따로 저장한다.
# Usage: python classical/pado_safe.py --config configs/TC1.json --inst 0
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
inst = int(sys.argv[sys.argv.index("--inst") + 1])
P_LAYERS, CONSTS, LR = cfg["p_layers"], cfg["consts"], cfg["lr"]          # 초기값: RZZ·RZ 각도 = +c, RX 각도 = −c
LWPP_STEPS, MAX_WEIGHT = cfg["lwpp"]["steps"], cfg["lwpp"]["max_weight"]   # LWPP: weight > max_weight인 Pauli 항을 버리는 근사
LWPP_PRESET, LWPP_DTYPE = cfg["lwpp"]["preset"], cfg["lwpp"]["dtype"]
THRESHOLD = cfg["distill_threshold"]                                       # 증류: |θ| <= THRESHOLD인 게이트를 회로에서 제거
EXACT_STEPS, EXACT_DEVICE = cfg["exact_steps"], cfg["exact_device"]
torch.set_default_dtype(torch.float64)  # PennyLane의 torch 인터페이스는 결과를 torch 기본 dtype으로 돌려준다

# ---- 문제 (전수 조사 해 E_min, E_max 포함)
problem_path = REPO / "problems" / f"{cfg['grid']}_inst{inst}_seed{cfg['seed_base'] + inst}.json"
prob = json.loads(problem_path.read_text())
n, e_min, e_max = prob["n_qubits"], prob["e_min"], prob["e_max"]
out = new_run_dir(cfg["name"], problem_path, "pado_safe")
tag = f"[{cfg['name']} {prob['name']} pado_safe]"

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

theta_init = np.zeros((len(CONSTS), n_params))
for i, c in enumerate(CONSTS):
    for g in gates[n:]:
        theta_init[i, g["param"]] = -c if g["gate"] == "RX" else c

# ---- 1/3 사전학습: PADO LWPP 대리 모델 (한 번 compile, 초기값 5종을 배치로 동시에 학습)
print(f"{tag} 1/3 LWPP 사전학습: max_weight {MAX_WEIGHT}, {LWPP_STEPS} step, {LWPP_DTYPE} (고전 GPU, 양자 자원 0)", flush=True)
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
n_terms = int(program.psum_union.x_mask.shape[0])  # compile 뒤 남은 Pauli 항 수 (zero-filter 이후)
print(f"{tag} compile {compile_s:.1f}s  terms={n_terms}  peak={compile_peak_gb:.2f} GB", flush=True)

thetas = torch.tensor(theta_init, device="cuda", requires_grad=True)
opt = torch.optim.Adam([thetas], lr=LR)
lwpp_energy = []
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for step in range(LWPP_STEPS):
    e = program.expvals(thetas, diff_mode="vjp")[:, 0]
    opt.zero_grad()
    e.sum().backward()  # sum: 초기값마다 자기 기울기만 받는다
    opt.step()
    lwpp_energy.append(e.detach().cpu().tolist())
    if step % 20 == 0 or step == LWPP_STEPS - 1:
        print(f"{tag} lwpp step {step:4d}  E_lwpp={[round(x, 4) for x in lwpp_energy[-1]]}", flush=True)
torch.cuda.synchronize()
lwpp_train_s = time.perf_counter() - t0
lwpp_train_peak_gb = torch.cuda.max_memory_allocated() / 1e9
theta_lwpp = thetas.detach().cpu().numpy()
lwpp_energy_final = program.expvals(torch.tensor(theta_lwpp, device="cuda"))[:, 0].cpu().tolist()
print(f"{tag} 사전학습 끝 {lwpp_train_s:.1f}s  peak={lwpp_train_peak_gb:.2f} GB", flush=True)

# ---- exact 에너지: 아무 게이트 목록(증류된 회로 포함)에 대해 PennyLane 상태벡터(GPU), adjoint 미분
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
    # 증류 전, LWPP 최종 파라미터의 exact 에너지 (기록용)
    exact_lwpp_only = float(energy(torch.tensor(theta_lwpp[i]), gates))

    # ---- 2/3 증류: |θ| <= THRESHOLD인 회전 게이트 제거, 남은 각도를 0..n_active-1로 다시 번호 매김
    active = [kk for kk in range(n_params) if abs(theta_lwpp[i, kk]) > THRESHOLD]
    new_idx = {old: new for new, old in enumerate(active)}
    pruned = [g for g in gates if g["gate"] == "H"]
    for g in gates[n:]:
        if g["param"] in new_idx:
            pruned.append({"gate": g["gate"], "wires": g["wires"], "param": new_idx[g["param"]]})
    print(f"{tag} 2/3 증류 c={c}: |θ| <= {THRESHOLD} 제거 → 남은 각도 {len(active)}/{n_params}"
          + ("  (모두 제거됨: 에너지가 상수)" if not active else ""), flush=True)

    # ---- 3/3 exact 미세조정: 증류된 회로에서. step t는 갱신 t회 뒤의 에너지
    print(f"{tag} 3/3 exact 미세조정 c={c}: {EXACT_STEPS} step (step당 환산 회로 {2 * len(active) + 1}개)", flush=True)
    theta = torch.tensor(theta_lwpp[i, active], requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    energies, traj, step_s = [], [], []
    for t in range(EXACT_STEPS + 1):
        t0 = time.perf_counter()
        opt.zero_grad()
        e = energy(theta, pruned)
        energies.append(float(e.detach()))
        traj.append(theta.detach().tolist())
        if t < EXACT_STEPS and active:  # 증류로 각도가 하나도 안 남으면 학습할 것이 없다
            e.backward()
            opt.step()
        step_s.append(time.perf_counter() - t0)
        print(f"{tag} c={c}  step {t:3d}  E={energies[-1]:+.6f}  AR={(e_max - energies[-1]) / (e_max - e_min):.4f}"
              f"  {step_s[-1]:.2f}s", flush=True)

    theta_final_full = np.zeros(n_params)  # 제거된 각도는 0(항등)이므로 전체 회로에 넣어도 같은 상태
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
    save_qasm(out / "qasm" / f"pado_safe_c{c}_init.qasm", n, gates, theta_init[i])
    save_qasm(out / "qasm" / f"pado_safe_c{c}_exact_start.qasm", n, pruned, traj[0])
    save_qasm(out / "qasm" / f"pado_safe_c{c}_final.qasm", n, pruned, traj[-1])

# ---- 저장
save_json(out / "circuit.json", {
    "n_qubits": n, "p_layers": P_LAYERS, "n_params": n_params,
    "convention": "R_P(theta) = exp(-i theta P / 2); RZZ = IsingZZ, initial state |0..0>; observable = sum J ZZ + sum h Z",
    "gates": gates, "edges": prob["edges"], "fields": prob["fields"],
    "distilled": "per init: keep gates whose param is in params.json inits[i].active_params, renumbered 0..n_active-1 in order",
})
save_json(out / "params.json", {
    "run": "pado_safe", "status": "ok", "experiment": cfg["name"], "problem": prob["name"], "e_min": e_min, "e_max": e_max,
    "config": {"p_layers": P_LAYERS, "consts": CONSTS, "lwpp_steps": LWPP_STEPS, "max_weight": MAX_WEIGHT,
               "lwpp_preset": LWPP_PRESET, "lwpp_dtype": LWPP_DTYPE, "threshold": THRESHOLD,
               "exact_steps": EXACT_STEPS, "lr": LR, "exact_device": EXACT_DEVICE, "diff_method": "adjoint"},
    "versions": env_info(),
    "lwpp": {"compile_s": compile_s, "compile_peak_gb": compile_peak_gb, "n_terms": n_terms,
             "train_s": lwpp_train_s, "train_peak_gb": lwpp_train_peak_gb},
    "inits": runs,
})
print(f"{tag} 끝: AR(0) → AR({EXACT_STEPS}) = " + ", ".join(f"{r['ar'][0]:.3f}→{r['ar'][-1]:.3f}" for r in runs)
      + f"  저장 {out.relative_to(REPO)}", flush=True)
