# standard 방식 연장 실행: 같은 초기값에서 PADO 방식의 최종 AR(A*)에 닿을 때까지 exact 학습을 계속한다 (상한 standard_long_max_steps).
# 회로·초기값·옵티마이저가 standard.py와 같으므로 처음 exact_steps step은 그 실행을 그대로 되풀이한다.
# Usage: python classical/standard_long.py --config configs/TC1.json --inst 0      (pado_safe 결과가 먼저 있어야 한다)
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from common import REPO, device_tag, env_info, new_run_dir, save_json

cfg = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
inst = int(sys.argv[sys.argv.index("--inst") + 1])
P_LAYERS, CONSTS, LR = cfg["p_layers"], cfg["consts"], cfg["lr"]          # 초기값: RZZ·RZ 각도 = +c, RX 각도 = −c
MAX_STEPS, EXACT_DEVICE = cfg["standard_long_max_steps"], cfg["exact_device"]   # 목표 AR에 못 닿으면 여기서 멈춘다
torch.set_default_dtype(torch.float64)  # PennyLane의 torch 인터페이스는 결과를 torch 기본 dtype으로 돌려준다

# ---- 문제와 초기값별 목표: PADO 방식이 exact 미세조정 끝에 도달한 AR
problem_path = REPO / "problems" / f"{cfg['grid']}_inst{inst}_seed{cfg['seed_base'] + inst}.json"
prob = json.loads(problem_path.read_text())
n, e_min, e_max = prob["n_qubits"], prob["e_min"], prob["e_max"]
pado = json.loads((REPO / f"results_on_{device_tag()}" / cfg["name"] / problem_path.stem / "pado_safe" / "params.json").read_text())
target = {run["c"]: run["ar"][-1] for run in pado["inits"]}
out = new_run_dir(cfg["name"], problem_path, "standard_long")
tag = f"[{cfg['name']} {prob['name']} standard_long]"
print(f"{tag} 시작: 초기값마다 목표 AR = PADO 최종 AR, 상한 {MAX_STEPS} step", flush=True)

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


# ---- 학습: 초기값마다 따로. step t는 갱신 t회 뒤의 에너지. AR >= 목표가 되는 첫 t에서 멈춘다
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
        if t % 50 == 0:
            print(f"{tag} c={c}  step {t:4d}  AR={ar[-1]:.4f}  목표={target[c]:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    wall_s = time.perf_counter() - t0
    print(f"{tag} c={c}: " + (f"step {hit}에서 목표 도달" if hit is not None else f"{MAX_STEPS} step 안에 미도달") + f"  ({wall_s:.1f}s)", flush=True)
    runs.append({
        "label": f"const_{c}", "c": c, "n_active": n_params, "target_ar": target[c], "steps_to_target": hit,
        "theta_init": theta_init.tolist(), "theta_final": theta.detach().tolist(),
        "energy": energies, "ar": ar, "wall_s": wall_s,
    })

# ---- 저장
save_json(out / "params.json", {
    "run": "standard_long", "status": "ok", "experiment": cfg["name"], "problem": prob["name"], "e_min": e_min, "e_max": e_max,
    "config": {"p_layers": P_LAYERS, "consts": CONSTS, "max_steps": MAX_STEPS, "lr": LR, "exact_device": EXACT_DEVICE,
               "diff_method": "adjoint", "circuits_per_step": 2 * n_params + 1,
               "target": "per init: final AR of the pado_safe run (pado_safe/params.json inits[i].ar[-1])"},
    "versions": env_info(),
    "inits": runs,
})
print(f"{tag} 끝: 도달 {sum(r['steps_to_target'] is not None for r in runs)}/{len(runs)}  저장 {out.relative_to(REPO)}", flush=True)
