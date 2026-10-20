# 이 폴더 스크립트들의 공용 함수: 장치 태그, 환경 정보, 결과 폴더, JSON 저장, QASM 저장.
import json
import platform
import re
import shutil
from pathlib import Path

import torch
from qiskit import QuantumCircuit, qasm2

REPO = Path(__file__).resolve().parents[1]


def device_tag():
    # "AMD Instinct MI300X VF" -> "MI300X"
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    m = re.search(r"\bMI\d{3}[A-Z]*\b", name, flags=re.IGNORECASE)
    return m.group(0).upper() if m else re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def env_info():
    # 결과 파일에 남기는 환경 정보 (패키지 버전, GPU)
    try:
        import hipauli as padopauli
    except ImportError:
        import padopauli
    import pennylane
    import qiskit
    return {"python": platform.python_version(), "torch": torch.__version__, "padopauli": padopauli.__version__,
            "pennylane": pennylane.__version__, "qiskit": qiskit.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def new_run_dir(experiment, problem_path, run):
    # results_on_<TAG>/<experiment>/<problem name>/<run>/ ; 끝난 실행은 덮어쓰지 않는다 (이름이 _smoke로 끝나는 실험만 예외).
    # 문제 파일(problem.json, problem.png)은 문제 폴더에 한 번만 복사한다 (실행 폴더마다 두지 않는다)
    problem_path = Path(problem_path)
    d = REPO / f"results_on_{device_tag()}" / experiment / problem_path.stem / run
    if (d / "params.json").exists() and not experiment.endswith("_smoke"):
        raise SystemExit(f"{d} already has a finished run; results are never overwritten")
    (d / "qasm").mkdir(parents=True, exist_ok=True)
    if not (d.parent / "problem.json").exists():
        shutil.copy(problem_path, d.parent / "problem.json")
        shutil.copy(problem_path.with_suffix(".png"), d.parent / "problem.png")
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def save_qasm(path, n_qubits, gates, theta):
    # 각도를 대입하고 Z 기저 측정을 붙인 OpenQASM 2.0 (다른 하드웨어에서 다시 돌릴 때 사용)
    qc = QuantumCircuit(n_qubits)
    for g in gates:
        if g["gate"] == "H":
            qc.h(g["wires"][0])
        elif g["gate"] == "RZZ":
            qc.rzz(float(theta[g["param"]]), *g["wires"])
        elif g["gate"] == "RZ":
            qc.rz(float(theta[g["param"]]), g["wires"][0])
        elif g["gate"] == "RX":
            qc.rx(float(theta[g["param"]]), g["wires"][0])
    qc.measure_all()
    Path(path).write_text(qasm2.dumps(qc))
