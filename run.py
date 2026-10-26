# 시험 실행기: configs/<TC>.json 하나로 그 시험의 모든 단계를 순서대로 돌린다. 지표는 metric.py가 계산한다.
# Usage: source <Pado-Lab-TT>/env.sh && python <repo>/run.py --exp TC1        (TC1, TC2, TC1_smoke)
# env.sh가 cwd를 tt-metal로 옮긴다. NPU를 여는 하위 스크립트는 그 폴더에서 돌아야 한다 (tt-metal 커널 소스가 상대경로)
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "classical"))
from common import device_tag, env_info  # noqa: E402

if "TT_METAL_HOME" not in os.environ or Path.cwd() != Path(os.environ["TT_METAL_HOME"]).resolve():
    raise SystemExit("Pado-Lab-TT의 env.sh를 먼저 source 한다 (TT_METAL_HOME, PYTHONPATH, cwd = tt-metal)")
exp = sys.argv[sys.argv.index("--exp") + 1]
config = str(REPO / "configs" / f"{exp}.json")
cfg = json.loads(Path(config).read_text())
root = REPO / f"results_on_{device_tag()}" / cfg["name"]
root.mkdir(parents=True, exist_ok=True)
PY = sys.executable
T_ALL = time.perf_counter()
print(f"===== {cfg['name']}: {cfg['title']}", flush=True)

# ---- 환경 기록: 패키지 목록과 NPU. editable 설치 경로의 홈 디렉터리는 ~로 적는다 (결과는 git에 들어간다)
print(f"\n===== [{cfg['name']}] 환경 기록 → {root.relative_to(REPO)}/env.txt", flush=True)
pip = subprocess.run([PY, "-m", "pip", "list"], capture_output=True, text=True).stdout.replace(str(Path.home()), "~")
(root / "env.txt").write_text(json.dumps(env_info(), indent=2) + "\n\n" + pip)
print(json.dumps(env_info()), flush=True)

# ---- 문제: 생성 스크립트를 다시 돌려 전수 조사 결과가 저장된 문제 파일과 일치하는지 확인 (다르면 멈춘다)
if not cfg["name"].endswith("_smoke"):
    print(f"\n===== [{cfg['name']}] 문제 파일 확인: problems/make_problems.py --grid {cfg['grid']} 재실행 (수 초)", flush=True)
    subprocess.run([PY, str(REPO / "problems" / "make_problems.py"), "--grid", cfg["grid"]], check=True)

# ---- PADO-Pauli(절단 없음)와 상태벡터의 에너지·기울기 일치 검사
print(f"\n===== [{cfg['name']}] PADO-Pauli vs PennyLane 일치 검사 (3x3, 호스트 float64 차이 1e-8 이내여야 PASS, NPU float32 차이는 출력만)", flush=True)
subprocess.run([PY, str(REPO / "classical" / "check_pado_vs_pennylane.py")], check=True)

# ---- 인스턴스마다 standard → pado_safe → standard_long(연장 실행, config의 run_standard_long이 true일 때만).
#      끝난 실행(params.json 있음)은 건너뛴다
runs = ["standard", "pado_safe"] + (["standard_long"] if cfg.get("run_standard_long", True) else [])
if "standard_long" not in runs:
    print(f"      (run_standard_long = false: 연장 실행은 하지 않는다. 지표의 도달 쌍·절감률 R은 비게 된다)", flush=True)
for inst in cfg["instances"]:
    problem = f"{cfg['grid']}_inst{inst}_seed{cfg['seed_base'] + inst}"
    for run in runs:
        done = (root / problem / run / "params.json").exists() and not cfg["name"].endswith("_smoke")
        print(f"\n===== [{cfg['name']}] {problem} / {run}" + ("  (이미 있음, 건너뜀)" if done else "")
              + (f"  (상한 {cfg['standard_long_max_steps']} step)" if run == "standard_long" and not done else ""), flush=True)
        if done:
            continue
        t0 = time.perf_counter()
        subprocess.run([PY, str(REPO / "classical" / f"{run}.py"), "--config", config, "--inst", str(inst)], check=True)
        print(f"      ({time.perf_counter() - t0:.0f}초)", flush=True)

print(f"\n===== [{cfg['name']}] 실행 끝 ({(time.perf_counter() - T_ALL) / 60:.1f}분). 다음: python {REPO / 'metric.py'} --exp {exp}", flush=True)
