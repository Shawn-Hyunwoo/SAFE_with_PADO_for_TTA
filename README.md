# SAFE with PADO — TTA 시험 재현 레포

PADO-Pauli(`padopauli` 또는 `hipauli`)의 LWPP 대리 모델로 사전학습·증류한 뒤 exact 미세조정하는 학습 방식(PADO 방식)이, 처음부터 exact 학습만 하는 방식(standard 방식)보다 양자 자원을 얼마나 아끼는지 잰다. 시험 항목 TC1·TC2·TC3의 정의와 판정 기준은 시험합의서를 따르며, 같은 값이 `configs/TC*.json`에 들어 있다.

- PADO 방식 (`pado_safe`): LWPP 사전학습(고전 GPU, 양자 자원 0) → |θ| ≤ 0.3인 게이트를 회로에서 제거(증류) → 남은 각도만 exact 미세조정
- standard 방식 (`standard`): 같은 초기값에서 exact 학습만. `standard_long`은 PADO 방식의 최종 AR에 닿을 때까지 연장한 실행
- exact 평가: 고전 시뮬레이션(TC1, TC2)은 PennyLane 상태벡터(adjoint 미분), 양자컴퓨터(TC3)는 parameter-shift
- 양자 자원: exact step T까지의 회로 수 C(T) = T × (2·N_act + 1) + 1. 고전 시뮬레이션에서는 환산값, TC3에서는 실제 제출한 회로 수
- LWPP(Low-Weight Pauli Propagation)는 weight가 `max_weight`를 넘는 Pauli 항을 버리는 근사 Pauli 전파다. 대리 모델의 에너지는 지표에 쓰지 않고, 모든 AR은 exact 평가로 계산한다.

방법 출처: Kim & Lee, "SAFE ma-QAOA" (https://arxiv.org/abs/2605.23377).

## 문제와 양자 회로 (2D Ising & ma-QAOA)

2차원 열린 경계($L \times L$) 격자 위에서 최근접(NN)·차근접 대각선(NNN) 상호작용과 세로 자기장을 갖는 **2D 스핀 글래스 이징(Ising) 바닥상태 탐색 문제**를 푼다.

$$H = \sum_{(i,j) \in E_{\text{NN}} \cup E_{\text{NNN}}} J_{ij} Z_i Z_j + \sum_{i=0}^{n-1} h_i Z_i$$

- **계수 분포**: 최근접 결합 $J_{ij}^{\text{NN}} \sim \mathcal{N}(0, 1^2)$, 대각선 차근접 결합 $J_{ij}^{\text{NNN}} \sim \mathcal{N}(0, 0.3^2)$, 국소 자기장 $h_i \sim \mathcal{N}(0, 0.1^2)$ (`seed = 120490 + inst`)
- **목표**: 바닥상태 에너지 $E_{\min} = \min_{z \in \{\pm 1\}^n} E(z)$에 가까워지도록 근사비 $\text{AR}(\boldsymbol{\theta}) = \frac{E_{\max} - \langle \psi(\boldsymbol{\theta}) | H | \psi(\boldsymbol{\theta}) \rangle}{E_{\max} - E_{\min}} \in [0, 1]$를 최대화 ($E_{\min}, E_{\max}$는 $2^n$개 스핀 배치 전수 조사로 계산)

```text
  (0) ─── (1) ─── (2)      ───  : 최근접 결합 J_ij^NN  ~ N(0, 1.0^2)
   │ ╲   ╱ │ ╲   ╱ │       ╲ ╱  : 대각선 결합 J_ij^NNN ~ N(0, 0.3^2)
   │   ╳   │   ╳   │       (q)  : 큐비트 q의 국소 자기장 h_q ~ N(0, 0.1^2)
   │ ╱   ╲ │ ╱   ╲ │
  (3) ─── (4) ─── (5)      * 3x3 격자 (n=9):  NN 12개 + NNN  8개 = 간선 20개, 필드 9개
   │ ╲   ╱ │ ╲   ╱ │       * 5x5 격자 (n=25): NN 40개 + NNN 32개 = 간선 72개, 필드 25개
   │   ╳   │   ╳   │
   │ ╱   ╲ │ ╱   ╲ │
  (6) ─── (7) ─── (8)
```

**회로 (ma-QAOA, 깊이 $p=3$)**: 초기 상태 $|+\rangle^{\otimes n} = H^{\otimes n}|0\rangle^{\otimes n}$ 위에, 층($l=1,\dots,p$)마다 **모든 게이트가 독립 파라미터 $\theta_k$를 갖는** 다중각도 QAOA 회로를 쌓는다 ($R_P(\theta) = \exp(-i\theta P / 2)$).

$$|\psi(\boldsymbol{\theta})\rangle = \left[ \prod_{l=1}^{p} \left( \prod_{i=0}^{n-1} R_X(\theta_{l,i}^X) \right) \left( \prod_{i=0}^{n-1} R_Z(\theta_{l,i}^Z) \right) \left( \prod_{(i,j) \in E} R_{ZZ}(\theta_{l,ij}^{ZZ}) \right) \right] H^{\otimes n} |0\rangle^{\otimes n}$$

- **층당 파라미터 수**: $|E|$개($R_{ZZ}$) + $n$개($R_Z$) + $n$개($R_X$) → **3x3($p=3$)은 총 114개**, **5x5($p=3$)는 총 366개**
- **PADO 증류**: LWPP 사전학습 후 $|\theta_k| \le 0.3$인 회전 게이트($R_{ZZ}, R_Z, R_X$)는 항등 연산($I$)에 가까우므로 회로에서 통째로 제거하고, 살아남은 $N_{\text{act}}$개 게이트만으로 미세조정한다.

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
python run.py --exp TC3 && python metric.py --exp TC3      # 3x3 양자(기본값 backend_mode="fake"): FakeNighthawk 노이즈 시뮬레이션으로 끝까지 실행 (약 5분)
                                                           # 실제 QPU 실행은 configs/TC3.json의 backend_mode를 "ibm"으로 바꾼 뒤 quantum/train_qpu.ipynb 직접 실행 → python metric.py --exp TC3
python run.py --exp TC1_smoke && python metric.py --exp TC1_smoke   # 코드 점검용 축소판 (3x3, 약 1분)
```

`run.py`는 단계마다 `===== [TC1] ...` 줄을 찍는다: 환경 기록 → 문제 파일 확인(생성 스크립트를 다시 돌려 전수 조사 결과가 저장된 파일과 같은지 검사) → PADO-Pauli와 PennyLane의 에너지·기울기 일치 검사 → 인스턴스마다 `standard` → `pado_safe` → `standard_long`. 각 스크립트도 자기 단계(`1/3 LWPP 사전학습`, `2/3 증류`, `3/3 exact 미세조정`)와 step별 AR을 찍는다. 끝난 실행은 건너뛰므로 중단된 뒤 다시 실행하면 이어서 돈다. 완료된 결과 폴더는 덮어쓰지 않는다(`*_smoke`만 예외). 연장 실행은 config의 `run_standard_long`(true/false)으로 켜고 끄며, 상한은 `standard_long_max_steps`다(시험합의서대로 TC1 500, TC2 2000). 연장 실행을 끄면 도달 쌍 수와 절감률 R은 계산되지 않는다.

`metric.py`는 `configs/<TC>.json`의 `criteria`로 판정하고 기준마다 PASS/FAIL과 값을 출력한다.

**TC3 실행 안내 (`FakeNighthawk` 기본값 vs 실제 QPU 실행)**: `configs/TC3.json`의 `qpu.backend_mode` 기본값은 `"fake"`이므로, `python run.py --exp TC3`를 실행하면 GPU 시작점 생성(`quantum/prepare_start_points.py`) 뒤 `FakeNighthawk` 노이즈 모델 시뮬레이터로 `quantum/train_qpu.ipynb` 전체를 끝까지 자동 실행한다. 실제 양자컴퓨터(QPU)에서 실행하려면 `configs/TC3.json`의 `qpu.backend_mode`를 `"ibm"`으로 바꾸고(`qpu.ibm_backend`: `"ibm_phoenix"`), **`quantum/train_qpu.ipynb` 노트북을 직접 열어** API 키 셀에 키를 입력한 뒤 셀들을 순서대로 실행하고(실행 뒤 키 삭제, 비워 두면 저장된 IBM 계정 사용), 완료 후 터미널에서 `python metric.py --exp TC3`를 실행한다. 결과는 `results_on_<기기 이름>/TC3/`에 쌓이고 `metric.py --exp TC3`는 결과가 있는 backend를 모두 평가한다(`--backend 이름`으로 하나만). 노트북은 step마다 결과를 저장하므로 끊겨도 다시 실행하면 이어서 돈다.

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

## 지표와 판정 기준 (시험합의서 2.4 라, 시험규격서)

`metric.py`가 `configs/<TC>.json`의 산정식·`criteria`대로 계산해 `metrics/pairs.csv`(TC1·TC2 쌍별), `metrics/steps.csv`(TC3 step별), `metrics/criteria.json`(판정)에 쓴다. 모든 AR은 exact 평가로 계산하며 LWPP 대리 모델의 에너지는 지표에 쓰지 않는다.

### 측정 지표 (TC1·TC2, 쌍마다)

| 지표 | 정의 | 의미 | `pairs.csv` 열 |
|---|---|---|---|
| AR(T) | (E_max − E)/(E_max − E_min). E는 exact step T의 에너지(T = 0은 시작점), E_min·E_max는 전수 조사 | 해의 품질 (1 = 최적) | `AR_std_0`, `AR_std_20`, `AR_pado_0`, `AR_pado_20` |
| N, N_act | 전체 각도 수(5x5: 366, 3x3: 114), 증류 후 남은 각도 수 | 회로 크기 | `N`, `N_act` |
| D | N_act / N | 잔존 비율 (step당 회로 수 비율) | `D` |
| ΔAR(20) | AR_PADO(20) − AR_std(20) | 같은 exact step 예산에서의 품질 이득 | `delta_AR20` |
| A*, A_0 | A* = AR_PADO(20), A_0 = AR_PADO(0) (쌍마다 정함) | 목표 품질, 시작점 품질 | = `AR_pado_20`, `AR_pado_0` |
| τ(A) | AR ≥ A에 처음 도달하는 exact step 수. standard는 상한 T_max(TC1 500, TC2 2000) 안에 못 닿으면 미도달 | 목표 품질에 드는 비용 | `tau_pado_Astar`, `tau_std_Astar`, `tau_std_A0`, `reached` |
| C(T), S(T) | C(T) = T × (2·N_act + 1) + 1 (standard는 2·N + 1), S(T) = C(T) × 1024 | step T까지의 누적 회로 수, 환산 shots | `C_pado`, `C_std`, `shots_pado_1024`, `shots_std_1024` |
| R | 1 − C_PADO(τ_PADO(A*)) / C_std(τ_std(A*)). 미도달 쌍은 제외 | 같은 품질까지의 회로 절감률 | `R` |
| 고전 비용 | LWPP compile 시간, 사전학습 시간, GPU peak memory | 절감의 대가로 쓴 고전 자원 (기록) | `lwpp_compile_s`, `lwpp_train_s`, `lwpp_peak_gb` |
| Status | 실행의 정상 완료 여부 | 실행 가능성 | `status_standard`, `status_pado_safe`, `status_standard_long` |

### 측정 지표 (TC3, backend마다)

| 지표 | 정의 | 의미 | `steps.csv` / `criteria.json` 열 |
|---|---|---|---|
| AR^Q(T) | step T의 파라미터로 기기에서 측정한 에너지(회로당 `qpu.shots`)로 계산한 AR | 기기에서 잰 해의 품질 | `AR` / `AR_pado_0`, `AR_pado_last`, `AR_std_0`, `AR_std_last` |
| AR^noiseless(T) | 같은 θ의 상태벡터 AR | 노이즈 없는 기준 | `AR_noiseless` |
| G(T) | abs(AR^Q(T) − AR^noiseless(T)) | 노이즈 격차 (기록, 판정 대상 아님) | `G` / `G_max_pado`, `G_max_std` |
| C_PADO, C_std | 실제 제출한 회로 수 = T × (2·N_act + 1) + 1. 증류 전 측정 1회는 제외 | 실측 양자 자원 | `n_circuits` / `C_pado`, `C_std`, `shots_pado`, `shots_std` |
| R_C | C_PADO / C_std | 회로 비율 | `R_C` |
| 기록 | 기기명, 캘리브레이션 일자, 큐비트 배치, job id, usage | 실행 조건 | `backend.json`, `job_id`, `usage_s` |

### 판정 기준 (`configs/<TC>.json`의 `criteria`, 결과는 `criteria.json`의 `checks`)

| TC | # | 판정 기준 | `criteria` 키 | 기준 |
|---|---|---|---|---|
| TC1 | 1 | 25쌍 모두 ΔAR(20) > 0 | `delta_ar20_min` | > 0 |
| TC1 | 2 | ΔAR(20)의 중앙값 ≥ 0.10 | `delta_ar20_median` | ≥ 0.10 |
| TC1 | 3 | AR_PADO(0) ≥ AR_std(20)인 쌍이 20쌍 이상 | `n_pairs_pado_start_ge_std_last` | ≥ 20 |
| TC1 | 4 | 25쌍 모두 D ≤ 0.7 | `D_max` | ≤ 0.7 |
| TC1 | 5 | standard가 T_max = 500 안에 A*에 도달한 쌍이 15쌍 이상 | `n_reached` | ≥ 15 |
| TC1 | 6 | 도달한 쌍의 R 중앙값 ≥ 0.80, 최솟값 ≥ 0.60 | `R_median`, `R_min` | ≥ 0.80, ≥ 0.60 |
| TC2 | 1 | 25쌍 모두 ΔAR(20) > 0 | `delta_ar20_min` | > 0 |
| TC2 | 2 | standard가 T_max = 2000 안에 A*에 도달한 쌍이 20쌍 이상 | `n_reached` | ≥ 20 |
| TC2 | 3 | 도달한 쌍의 R 중앙값 ≥ 0.80, 최솟값 ≥ 0.60 | `R_median`, `R_min` | ≥ 0.80, ≥ 0.60 |
| TC2 | 4 | 25쌍 모두 τ_std(A_0) ≥ 20 | `tau_std_A0_min` | ≥ 20 |
| TC3 | 1 | AR^Q_PADO(T_PADO) ≥ AR^Q_std(T_std) | `AR_pado_last_minus_AR_std_last` | ≥ 0 |
| TC3 | 2 | R_C ≤ 0.20 | `R_C` | ≤ 0.20 |
| TC3 | 3 | AR^Q_PADO(0) ≥ AR^Q_std(T_std) | `AR_pado_0_minus_AR_std_last` | ≥ 0 |

TC3는 backend(FakeNighthawk, IBM 실기기)마다 따로 판정하며, T_PADO·T_std는 `qpu.steps`(현재 3, 12)다. `criteria` 키 끝의 `_gt`/`_ge`/`_le`가 비교 방향이고, `metric.py`는 기준마다 값·임계값·PASS/FAIL을 출력한다. TC마다 모든 기준을 만족하면 전체 PASS.

참고: PennyLane의 torch 인터페이스는 결과를 torch 기본 dtype으로 돌려주므로 스크립트 첫머리에서 `torch.set_default_dtype(torch.float64)`로 설정한다. 이 설정이 없으면 exact 에너지가 float32로 반올림된다.
