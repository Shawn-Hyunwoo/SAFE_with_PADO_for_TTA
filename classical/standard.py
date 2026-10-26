# standard 방식: 상수 초기값에서 exact(상태벡터) 학습만 exact_steps 회. 초기값 c마다 따로 돌고 step별 에너지·AR·θ를 저장한다.
# Usage: python classical/standard.py --config configs/TC1.json --inst 0
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from common import REPO, env_info, new_run_dir, save_json

cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
inst = int(sys.argv[sys.argv.index("--inst") + 1])
P_LAYERS, CONSTS, LR = cfg["p_layers"], cfg["consts"], cfg["lr"]          # 초기값: RZZ·RZ 각도 = +c, RX 각도 = −c
EXACT_STEPS, EXACT_DEVICE = cfg["exact_steps"], cfg["exact_device"]
torch.set_default_dtype(torch.float64)  # PennyLane의 torch 인터페이스는 결과를 torch 기본 dtype으로 돌려준다

# ---- 문제 (전수 조사 해 E_min, E_max 포함)
problem_path = REPO / "problems" / f"{cfg['grid']}_inst{inst}_seed{cfg['seed_base'] + inst}.json"
prob = json.loads(problem_path.read_text())
n, e_min, e_max = prob["n_qubits"], prob["e_min"], prob["e_max"]
out = new_run_dir(cfg["name"], problem_path, "standard")
tag = f"[{cfg['name']} {prob['name']} standard]"
print(f"{tag} 시작: 초기값 {len(CONSTS)}종 × exact {EXACT_STEPS} step ({EXACT_DEVICE}, adjoint 미분)", flush=True)

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
save_json(out / "circuit.json", {
    "n_qubits": n, "p_layers": P_LAYERS, "n_params": n_params,
    "convention": "R_P(theta) = exp(-i theta P / 2); RZZ = IsingZZ, initial state |0..0>; observable = sum J ZZ + sum h Z",
    "gates": gates, "edges": prob["edges"], "fields": prob["fields"],
})

# ---- exact 에너지: PennyLane 상태벡터(CPU), adjoint 미분
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


# ---- 학습: 초기값마다 따로. step t는 갱신 t회 뒤의 에너지 (t = EXACT_STEPS가 최종)
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
        print(f"{tag} c={c}  step {t:3d}  E={energies[-1]:+.6f}  AR={(e_max - energies[-1]) / (e_max - e_min):.4f}"
              f"  {step_s[-1]:.2f}s", flush=True)
    ar = [(e_max - x) / (e_max - e_min) for x in energies]
    runs.append({
        "label": f"const_{c}", "c": c, "n_active": n_params, "active_params": list(range(n_params)),
        "theta_init": theta_init.tolist(), "theta_final": traj[-1],
        "energy": energies, "ar": ar, "theta_traj": traj, "step_wall_s": step_s,
    })

# ---- 저장
save_json(out / "params.json", {
    "run": "standard", "status": "ok", "experiment": cfg["name"], "problem": prob["name"], "e_min": e_min, "e_max": e_max,
    "config": {"p_layers": P_LAYERS, "consts": CONSTS, "exact_steps": EXACT_STEPS, "lr": LR,
               "exact_device": EXACT_DEVICE, "diff_method": "adjoint", "circuits_per_step": 2 * n_params + 1},
    "versions": env_info(),
    "inits": runs,
})
print(f"{tag} 끝: AR(0) → AR({EXACT_STEPS}) = " + ", ".join(f"{r['ar'][0]:.3f}→{r['ar'][-1]:.3f}" for r in runs)
      + f"  저장 {out.relative_to(REPO)}", flush=True)
