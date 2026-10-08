# Shared by the scripts of this folder only: device tag, result folder, JSON save, circuit/QASM export.
import json
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


def new_run_dir(experiment, problem_path, arm):
    # results_on_<TAG>/<experiment>/<problem name>/<arm>/ ; refuses to overwrite a finished run (smoke runs excepted)
    problem_path = Path(problem_path)
    d = REPO / f"results_on_{device_tag()}" / experiment / problem_path.stem / arm
    if (d / "params.json").exists() and not experiment.endswith("_smoke"):
        raise SystemExit(f"{d} already has a finished run; results are never overwritten")
    (d / "qasm").mkdir(parents=True, exist_ok=True)
    shutil.copy(problem_path, d / "problem.json")
    shutil.copy(problem_path.with_suffix(".png"), d / "problem.png")
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def save_qasm(path, n_qubits, gates, theta):
    # OpenQASM 2.0 with angles bound and a final Z-basis measurement (for later hardware runs)
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
