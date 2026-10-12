# 시험 실행기: configs/<TC>.json 하나로 그 시험의 모든 단계를 순서대로 돌린다. 지표는 metric.py가 계산한다.
# Usage: python run.py --exp TC1        (TC1, TC2, TC3, TC1_smoke)
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "classical"))
from common import device_tag, env_info  # noqa: E402

exp = sys.argv[sys.argv.index("--exp") + 1]
cfg = json.loads((REPO / "configs" / f"{exp}.json").read_text())
root = REPO / f"results_on_{device_tag()}" / cfg["name"]
root.mkdir(parents=True, exist_ok=True)
PY = sys.executable
T_ALL = time.perf_counter()
print(f"===== {cfg['name']}: {cfg['title']}", flush=True)

# ---- 환경 기록: 패키지 목록과 GPU
print(f"\n===== [{cfg['name']}] 환경 기록 → {root.relative_to(REPO)}/env.txt", flush=True)
pip = subprocess.run([PY, "-m", "pip", "list"], capture_output=True, text=True).stdout
(root / "env.txt").write_text(json.dumps(env_info(), indent=2) + "\n\n" + pip)
print(json.dumps(env_info()), flush=True)

# ---- 문제: 생성 스크립트를 다시 돌려 전수 조사 결과가 저장된 문제 파일과 일치하는지 확인 (다르면 멈춘다)
if not cfg["name"].endswith("_smoke"):
    print(f"\n===== [{cfg['name']}] 문제 파일 확인: problems/make_problems.py --grid {cfg['grid']} 재실행 (5x5는 약 3분, 3x3은 수 초)", flush=True)
    subprocess.run([PY, "problems/make_problems.py", "--grid", cfg["grid"]], cwd=REPO, check=True)

if cfg["kind"] == "classical":
    # ---- PADO-Pauli(절단 없음)와 상태벡터의 에너지·기울기 일치 검사
    print(f"\n===== [{cfg['name']}] PADO-Pauli vs PennyLane 일치 검사 (3x3, 차이 1e-8 이내여야 PASS)", flush=True)
    subprocess.run([PY, "classical/check_pado_vs_pennylane.py"], cwd=REPO, check=True)

    # ---- 인스턴스마다 standard → pado_safe → standard_long. 끝난 실행(params.json 있음)은 건너뛴다
    for inst in cfg["instances"]:
        problem = f"{cfg['grid']}_inst{inst}_seed{cfg['seed_base'] + inst}"
        for run in ["standard", "pado_safe", "standard_long"]:
            done = (root / problem / run / "params.json").exists() and not cfg["name"].endswith("_smoke")
            print(f"\n===== [{cfg['name']}] {problem} / {run}" + ("  (이미 있음, 건너뜀)" if done else ""), flush=True)
            if done:
                continue
            t0 = time.perf_counter()
            subprocess.run([PY, f"classical/{run}.py", "--config", f"configs/{exp}.json", "--inst", str(inst)], cwd=REPO, check=True)
            print(f"      ({time.perf_counter() - t0:.0f}초)", flush=True)
else:
    # ---- 시작점(GPU) → 양자컴퓨터 학습 노트북 실행 (configs/<TC>.json의 qpu 설정을 읽는다)
    problem = f"{cfg['grid']}_inst{cfg['instance']}_seed{cfg['seed_base'] + cfg['instance']}"
    done = (root / problem / "start_points" / "params.json").exists()
    print(f"\n===== [{cfg['name']}] {problem} / start_points" + ("  (이미 있음, 건너뜀)" if done else ""), flush=True)
    if not done:
        subprocess.run([PY, "quantum/prepare_start_points.py", "--config", f"configs/{exp}.json"], cwd=REPO, check=True)
    q = cfg["qpu"]
    if q["backend_mode"] != "fake":
        # 실기기(IBM) 실행은 이 실행기가 하지 않는다. 사용자가 quantum/train_qpu.ipynb를 열어 키 셀을 채우고 직접 실행한다
        raise SystemExit(f"[{cfg['name']}] qpu.backend_mode = {q['backend_mode']!r}: 실기기 실행은 run.py가 하지 않는다. "
                         "quantum/train_qpu.ipynb를 직접 열어 실행한 뒤 python metric.py --exp TC3")
    print(f"\n===== [{cfg['name']}] 양자컴퓨터 학습(노이즈 모델 시뮬레이션): quantum/train_qpu.ipynb 실행 ({q['fake_backend']}, "
          f"{q['shots']} shots, PADO {q['steps']['pado_safe']} step / standard {q['steps']['standard']} step)", flush=True)
    subprocess.run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=-1", "train_qpu.ipynb"], cwd=REPO / "quantum", check=True)

print(f"\n===== [{cfg['name']}] 실행 끝 ({(time.perf_counter() - T_ALL) / 60:.1f}분). 다음: python metric.py --exp {exp}", flush=True)
