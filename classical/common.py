# 이 폴더 스크립트들의 공용 함수: 장치 태그, 환경 정보, 메모리 측정, 결과 폴더, JSON 저장.
import glob
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
CARD_TYPES = "/sys/class/tenstorrent/tenstorrent!*/tt_card_type"   # Tenstorrent 드라이버가 칩마다 두는 보드 이름 ("p300c")


def device_tag():
    # 보드 이름 "p300c" -> "P300C". sysfs만 읽고 장치는 열지 않는다
    cards = sorted(glob.glob(CARD_TYPES))
    return Path(cards[0]).read_text().strip().upper() if cards else "cpu"


def env_info():
    # 결과 파일에 남기는 환경 정보: 패키지 버전, 엔진(PADO-Lab)과 포트(Pado-Lab-TT)의 커밋, 칩마다 보드 이름
    import padopauli
    import pennylane
    git_head = ["git", "rev-parse", "--short", "HEAD"]
    pado_dir = Path(padopauli.__file__).resolve().parent
    tt_dir = Path(importlib.util.find_spec("padopauli_tt").origin).resolve().parent
    return {"python": platform.python_version(), "torch": torch.__version__,
            "padopauli": padopauli.__version__,
            "padopauli_commit": subprocess.run(git_head, cwd=pado_dir, capture_output=True, text=True).stdout.strip(),
            "padopauli_tt": importlib.metadata.version("padopauli-tt"),
            "padopauli_tt_commit": subprocess.run(git_head, cwd=tt_dir, capture_output=True, text=True).stdout.strip(),
            "ttnn": importlib.metadata.version("ttnn"), "pennylane": pennylane.__version__,
            "npu": [Path(p).read_text().strip() for p in sorted(glob.glob(CARD_TYPES))]}


def reset_peak_rss():
    # 호스트 프로세스의 peak RSS(VmHWM)를 지금 RSS로 되돌린다 (Linux /proc/self/clear_refs)
    Path("/proc/self/clear_refs").write_text("5")


def peak_rss_gb():
    # 마지막 reset_peak_rss() 뒤 호스트 프로세스의 peak RSS (VmHWM)
    return int(re.search(r"VmHWM:\s+(\d+) kB", Path("/proc/self/status").read_text()).group(1)) * 1024 / 1e9


def npu_dram_gb(dev):
    # 읽은 시점에 NPU DRAM에 할당된 양 (ttnn allocator, bank 전체). 최댓값이 아니다
    import ttnn
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return v.num_banks * v.total_bytes_allocated_per_bank / 1e9


def new_run_dir(experiment, problem_path, run):
    # results_on_<TAG>/<experiment>/<problem name>/<run>/ ; 끝난 실행은 덮어쓰지 않는다 (이름이 _smoke로 끝나는 실험만 예외).
    # 문제 파일(problem.json, problem.png)은 문제 폴더에 한 번만 복사한다 (실행 폴더마다 두지 않는다)
    problem_path = Path(problem_path)
    d = REPO / f"results_on_{device_tag()}" / experiment / problem_path.stem / run
    if (d / "params.json").exists() and not experiment.endswith("_smoke"):
        raise SystemExit(f"{d} already has a finished run; results are never overwritten")
    d.mkdir(parents=True, exist_ok=True)
    if not (d.parent / "problem.json").exists():
        shutil.copy(problem_path, d.parent / "problem.json")
        shutil.copy(problem_path.with_suffix(".png"), d.parent / "problem.png")
    return d


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))
