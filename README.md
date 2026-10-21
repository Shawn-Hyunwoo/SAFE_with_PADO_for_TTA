# SAFE with PADO — TTA 시험 재현 레포

PADO-Pauli(`padopauli` 또는 `hipauli`)의 LWPP 대리 모델로 사전학습·증류한 뒤 exact 미세조정하는 학습 방식(PADO 방식)이, 처음부터 exact 학습만 하는 방식(standard 방식)보다 양자 자원을 얼마나 아끼는지 잰다. 시험 항목 TC1·TC2·TC3의 정의와 판정 기준은 시험합의서를 따르며, 같은 값이 `configs/TC*.json`에 들어 있다.

- PADO 방식 (`pado_safe`): LWPP 사전학습(고전 GPU, 양자 자원 0) → |θ| ≤ 0.3인 게이트를 회로에서 제거(증류) → 남은 각도만 exact 미세조정
- standard 방식 (`standard`): 같은 초기값에서 exact 학습만. `standard_long`은 PADO 방식의 최종 AR에 닿을 때까지 연장한 실행
- exact 평가: 고전 시뮬레이션(TC1, TC2)은 PennyLane 상태벡터(adjoint 미분), 양자컴퓨터(TC3)는 parameter-shift
- 양자 자원: exact step T까지의 회로 수 C(T) = T × (2·N_act + 1) + 1. 고전 시뮬레이션에서는 환산값, TC3에서는 실제 제출한 회로 수
- LWPP(Low-Weight Pauli Propagation)는 weight가 `max_weight`를 넘는 Pauli 항을 버리는 근사 Pauli 전파다. 대리 모델의 에너지는 지표에 쓰지 않고, 모든 AR은 exact 평가로 계산한다.

방법 출처: Kim & Lee, "SAFE ma-QAOA" (https://arxiv.org/abs/2605.23377).

| | TC1 | TC2 | TC3 |
|---|---|---|---|
| 문제 | 5x5 격자(25큐비트), 인스턴스 5개 | 3x3 격자(9큐비트), 인스턴스 5개 | 3x3 격자, 인스턴스 0 |
| 초기값 | 상수 5종 (RZZ·RZ = +c, RX = −c, c = 0.01, 0.05, 0.1, 0.2, 0.4) | 같음 | c = 0.1 |
| LWPP 사전학습 | max_weight 5, 200 step | max_weight 3, 100 step | max_weight 3, 100 step |
| exact 단계 | PennyLane, 20 step. standard는 PADO 최종 AR까지 연장(상한 500) | PennyLane, 20 step. 연장 상한 2000 | 양자컴퓨터 parameter-shift, PADO 3 step / standard 12 step, 회로당 512 shots |
| 실행 환경 | GPU 서버 | GPU 서버 | FakeNighthawk(서버 로컬) 및 IBM 실기기 (`ibm_phoenix`) |

## 실행 (시험자용)

환경을 만든 뒤 시험 항목마다 두 줄이다. 결과는 `results_on_<GPU 이름>/<TC>/`에, 지표·판정·그림은 그 안의 `metrics/`에 쓴다.

```bash
conda activate <env>                                       # padopauli 또는 hipauli가 설치된 환경 활성화
python run.py --exp TC1 && python metric.py --exp TC1      # 5x5 고전: 약 1~1.5시간 (연장 실행이 상한 500 step까지 다 가면 최대 8시간)
python run.py --exp TC2 && python metric.py --exp TC2      # 3x3 고전: 약 10분
python run.py --exp TC3 && python metric.py --exp TC3      # 3x3 양자컴퓨터: FakeNighthawk 시뮬레이션 약 10분 (실기기는 아래)
python run.py --exp TC1_smoke && python metric.py --exp TC1_smoke   # 코드 점검용 축소판 (3x3, 약 1분)
```

`run.py`는 단계마다 `===== [TC1] ...` 줄을 찍는다: 환경 기록 → 문제 파일 확인(생성 스크립트를 다시 돌려 전수 조사 결과가 저장된 파일과 같은지 검사) → PADO-Pauli와 PennyLane의 에너지·기울기 일치 검사 → 인스턴스마다 `standard` → `pado_safe` → `standard_long`. 각 스크립트도 자기 단계(`1/3 LWPP 사전학습`, `2/3 증류`, `3/3 exact 미세조정`)와 step별 AR을 찍는다. 끝난 실행은 건너뛰므로 중단된 뒤 다시 실행하면 이어서 돈다. 완료된 결과 폴더는 덮어쓰지 않는다(`*_smoke`만 예외). 연장 실행은 config의 `run_standard_long`(true/false)으로 켜고 끄며, 상한은 `standard_long_max_steps`다(시험합의서대로 TC1 500, TC2 2000). 연장 실행을 끄면 도달 쌍 수와 절감률 R은 계산되지 않는다.

`metric.py`는 `configs/<TC>.json`의 `criteria`로 판정하고 기준마다 PASS/FAIL과 값을 출력한다.

**TC3 실기기**: `run.py --exp TC3`는 노이즈 모델 시뮬레이션(FakeNighthawk)까지만 돌린다. 실기기는 `configs/TC3.json`의 `qpu.backend_mode`를 `"ibm"`, `qpu.ibm_backend`에 기기 이름(예: `ibm_phoenix`)을 넣고, `quantum/train_qpu.ipynb`를 열어 키 셀에 API 키를 넣은 뒤 직접 실행한다(실행 뒤 키 삭제. 비워 두면 이 컴퓨터에 저장된 계정을 쓴다). 결과는 `results_on_<기기 이름>/TC3/`에 쌓이고 `metric.py --exp TC3`는 결과가 있는 backend를 모두 평가한다(`--backend 이름`으로 하나만). `qpu.rep_delay`(shot 사이 대기, 초)는 기기 허용 범위 안에서 정하고 `null`이면 기기 기본값이다. 노트북은 step마다 결과를 저장하므로 끊겨도 다시 실행하면 이어서 돈다.

## 환경 (AMD GPU, ROCm 7.2)

Python 3.11, AMD Instinct GPU(MI300X에서 검증)를 기준으로 한다. PADO-Pauli 대리 모델 엔진은 공개 PyPI 패키지(`padopauli 2.0.1`) 또는 ROCm C++/HIP 가속 패키지(`hipauli`) 두 환경 모두 지원하며, 스크립트는 실행 환경에 `hipauli`가 설치되어 있으면 이를 우선 사용하고 없으면 `padopauli`를 사용한다(동일 API·동일 수치 결과). 순서가 중요하다. ROCm torch를 먼저 설치해야 `padopauli`가 CUDA torch를 끌어오지 않는다.

```bash
conda create -n <env> python=3.11.15 -y
conda activate <env>
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt     # padopauli 2.0.1(PyPI), pennylane 0.45.1 + lightning-amdgpu 0.45.0, qiskit 2.5.1 ...
# (선택) hipauli 가속 환경에서는 hipauli 패키지가 설치된 환경을 그대로 활성화하여 실행 가능
cd /tmp && python -c "import torch, pennylane; p = __import__('hipauli') if __import__('importlib.util').util.find_spec('hipauli') else __import__('padopauli'); print(torch.__version__, torch.cuda.is_available(), p.__name__, p.__version__)"
```

## 구성

```
configs/                    시험 설정 (문제·회로·학습·판정 기준): TC1.json, TC2.json, TC3.json, TC1_smoke.json
run.py                      시험 실행기        python run.py --exp TC1
metric.py                   지표·판정·그림     python metric.py --exp TC1
problems/                   문제 생성 스크립트(make_problems.py), 인스턴스 JSON(전수 조사 해 포함), PNG
classical/                  TC1·TC2: standard.py, pado_safe.py, standard_long.py, check_pado_vs_pennylane.py, common.py
quantum/                    TC3: prepare_start_points.py(GPU, 두 방식의 시작점), train_qpu.ipynb(양자컴퓨터 학습), common.py
extras/                     시험 항목 밖의 참고 실험 (판정에 쓰지 않음)
  sampling_h2/              5x5 5개 시점의 비트열 샘플링 노트북 (노이즈 없는 PennyLane, Quantinuum H2)
  qpu_trial/                실기기 1 step 시운전 노트북 (시간·usage·배치 확인)
results_on_<장치>/<TC>/     결과 (예: results_on_MI300X/TC1/, results_on_fake_nighthawk/TC3/). git에는 넣지 않고 서버에만 둔다
results_on_<장치>/extras/   참고 실험 결과 (코드 폴더와 같은 이름: sampling_h2/, qpu_trial/)
```

## 결과 파일

- `results_on_<GPU>/TC1/<문제>/{standard,pado_safe,standard_long}/`
  - `params.json`: 설정, 패키지 버전, 초기값별 θ_init·θ_final, step별 θ·에너지·AR·시간. `pado_safe`는 LWPP compile·학습 시간과 GPU peak memory, 증류 뒤 남은 각도 번호(`active_params`), 증류 전 exact 에너지도. `standard_long`은 목표 AR과 도달 step.
  - `circuit.json`(게이트 목록, 각도 규약 R_P(θ) = exp(−iθP/2)), `qasm/`(초기·최종 파라미터를 넣고 측정을 붙인 OpenQASM 2.0).
  - 문제 파일 `problem.json/png`는 실행 폴더가 아니라 문제 폴더(`results_on_<GPU>/TC1/<문제>/`)에 하나만 둔다.
- `results_on_<GPU>/TC1/env.txt`: 패키지 목록과 GPU.
- `results_on_<backend>/TC3/<문제>/{standard,pado_safe}/params.json`: step별 측정 에너지·표준오차·AR, 같은 θ의 노이즈 없는 AR, 기울기, job id, IBM 사용 시간(usage)과 시각. `backend.json`(기기, 캘리브레이션 일자, rep_delay, 큐비트 배치, ISA 회로의 2큐비트 게이트 수·깊이). `pado_safe/pre_distill.json`(증류 전 측정 1회, 집계 제외). 시작점은 `results_on_<GPU>/TC3/<문제>/start_points/`.
- `metrics/` (metric.py): `pairs.csv`(TC1·TC2 쌍별 AR·τ·C·R·D·환산 shots), `steps.csv`(TC3 step별), `criteria.json`(값, 임계값, PASS/FAIL, 전체 판정, 실행 상태), `energy_curves*.png`, `summary_ar.png`.

## 지표 (metric.py, 시험합의서의 산정식)

- AR = (E_max − E)/(E_max − E_min). E_min·E_max는 전수 조사. AR(T)는 exact step T(갱신 T회 뒤)에서의 값, T = 0은 시작점.
- ΔAR(20) = AR_PADO(20) − AR_std(20). 잔존 비율 D = N_act / N.
- 목표 품질 A* = AR_PADO(20). τ(A) = AR ≥ A에 처음 닿는 step(상한 안에 못 닿으면 미도달). C = τ × (2N+1) + 1. 회로 절감률 R = 1 − C_PADO / C_std. τ_std(A_0)은 standard가 PADO의 시작 AR에 닿는 step.
- TC3: 같은 기기에서 잰 AR^Q, 실제 회로 수와 shots, R_C = C_PADO(3) / C_std(12), 노이즈 격차 G = |AR^Q − AR^noiseless|(기록).
- 판정 기준은 `configs/<TC>.json`의 `criteria`다. 키 끝의 `gt`/`ge`/`le`가 비교 방향이고, 결과는 `criteria.json`의 `checks`에 값과 함께 남는다.

참고: PennyLane의 torch 인터페이스는 결과를 torch 기본 dtype으로 돌려주므로 스크립트 첫머리에서 `torch.set_default_dtype(torch.float64)`로 설정한다. 이 설정이 없으면 exact 에너지가 float32로 반올림된다.
