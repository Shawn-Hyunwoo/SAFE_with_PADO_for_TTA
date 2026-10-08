# SAFE with PADO — TTA 평가용 재현 레포

SAFE(Surrogate-Assisted and Fine-tuning Enhanced) ma-QAOA가 처음부터 exact 최적화를 하는 방식보다 exact 자원을 얼마나 아끼는지 비교한다.

- **SAFE**: PADO-Pauli(`padopauli`)의 LWPP 대리 모델로 사전학습한 뒤, 크기가 작은 각도를 회로에서 제거(증류)하고, 남은 각도만 exact로 fine-tuning한다. LWPP는 weight가 `max_weight`를 넘는 Pauli 항을 버리는 근사 Pauli 전파다.
- **exact-only**: 같은 초기값에서 바로 exact로 학습한다.
- **exact 최적화**: 고전 시뮬레이션에서는 PennyLane 상태벡터의 adjoint 미분, 실기기에서는 parameter-shift이다.

비교 지표는 두 가지다.
1. 같은 exact step 수에서의 approximation ratio(AR)
2. 같은 AR에 도달하기까지 필요한 exact step 수와 회로 실행 수(QPU 사용량)

방법의 출처는 Kim & Lee, "SAFE ma-QAOA: Surrogate-Assisted and Fine-Tuning Enhanced Multi-Angle QAOA with Parameter Distillation" (https://arxiv.org/abs/2605.23377)이다.

| | 실험 1 (고전 5x5) | 실험 2 (고전 3x3) | 실험 3 (양자 3x3) |
|---|---|---|---|
| 문제 | 5x5 격자, 인스턴스 5개 | 3x3 격자, 인스턴스 5개 | 3x3 격자, 인스턴스 1개 |
| ansatz | ma-QAOA, p = 3 (366 각도) | ma-QAOA, p = 3 (114 각도) | ma-QAOA, p = 3 (114 각도) |
| 초기화 | 상수 5종 (γ = +c, β = −c, c = 0.01, 0.05, 0.1, 0.2, 0.4) | 상수 5종 (같음) | 상수 1종 (c = 0.1) |
| LWPP 사전학습 | max_weight 5, 200 step | max_weight 3, 100 step | max_weight 3, 100 step |
| 증류 | \|θ\| ≤ 0.3 제거 | \|θ\| ≤ 0.3 제거 | \|θ\| ≤ 0.3 제거 |
| exact 단계 | PennyLane `lightning.amdgpu`, 20 step | PennyLane `lightning.amdgpu`, 20 step (exact-only는 SAFE 최종 AR까지 연장 실행도 함) | Qiskit parameter-shift, SAFE 5 step / exact-only 20 step, 회로당 1024 shots |

공통 설정은 Adam lr 0.02(기본 betas)이고, LWPP는 `gpu` preset에 float32를 쓴다.

## 문제

H = Σ_i h_i Z_i + Σ_(i,j) J_ij Z_i Z_j이며 열린 경계 격자를 쓴다.
- 최근접 이웃(NN) J ~ N(0, 1), 대각 다음 이웃(NNN) J ~ N(0, 0.3²), h ~ N(0, 0.1²)
- seed = 120490 + 인스턴스 번호

`problems/make_problems.py`는 2^n개 스핀 배치를 모두 계산(brute force)해서 E_min, E_max, 바닥 상태 비트열을 JSON에 저장한다. 같은 이름의 PNG에 격자, 결합, 필드, 바닥 상태 스핀을 그린다. 다시 실행하면 기존 JSON과 같은지 확인하고, 다르면 멈춘다.

AR = (E_max − E) / (E_max − E_min)이다. 그림에서는 brute force E_min을 기준선으로 쓴다.

## 구성

```
problems/               문제 생성 스크립트, 인스턴스 JSON, PNG
exp1_classical_5x5/     실험 1 코드
  check_pado_vs_pennylane.py   PADO(절단 없음)와 PennyLane의 에너지·기울기 일치 확인 (3x3, 수 초)
  exact_only.py / safe.py      두 arm
  report.py                    표(summary.csv)와 그림(energy_curves.png)
exp2_classical_3x3/     실험 2 코드 (실험 1과 같은 코드, 크기·max_weight·LWPP step만 다름) — 실험 3의 고전 기준선
  exact_only.py / safe.py      두 arm
  exact_only_long.py           exact-only를 SAFE 최종 AR에 닿을 때까지 연장 (최대 2000 step)
  report.py                    표와 그림 (연장 실행 결과가 있으면 summary_long.csv, energy_curves_long.png도)
exp3_quantum_3x3/       실험 3 코드
  prepare_start_points.py      케이스 1개(인스턴스 0, c = 0.1)의 두 QPU arm 시작점 (SAFE 쪽은 LWPP → 증류, GPU에서 실행)
  train_qpu.ipynb              QPU 학습 노트북. 기본은 fake backend(FakeNighthawk) 노이즈 시뮬레이션,
                               BACKEND_MODE = "ibm"과 API 키 셀만 바꾸면 실기기. 9큐비트는 기기 안의 3x3 격자
                               (최근접 이웃 12쌍이 모두 직접 연결되고 오류가 가장 작은 배치)에 고정해 놓는다
results_on_<장치>/      실행 결과 (예: results_on_MI300X/)
```

## 환경 (AMD GPU, ROCm 7.2)

Python 3.11, AMD Instinct GPU(MI300X에서 검증)를 기준으로 한다. 순서가 중요하다. ROCm torch를 먼저 설치해야 `padopauli`가 CUDA torch를 끌어오지 않는다.

```bash
conda create -n <env> python=3.11.15 -y
conda activate <env>
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt     # padopauli 2.0.1(PyPI), pennylane 0.45.1 + lightning-amdgpu 0.45.0, qiskit 2.5.1 ...
cd /tmp && python -c "import torch, padopauli, pennylane; print(torch.__version__, torch.cuda.is_available(), padopauli.__version__)"
```

## 실행

```bash
python problems/make_problems.py                     # 인스턴스 + brute force (약 3분, 이미 레포에 포함)
python exp1_classical_5x5/check_pado_vs_pennylane.py # PASS가 나와야 함

# 실험 1: smoke (3x3, 수십 초, 결과는 results_on_<장치>/exp1_smoke/)
python exp1_classical_5x5/exact_only.py problems/grid3x3_inst0_seed120490.json 2
python exp1_classical_5x5/safe.py problems/grid3x3_inst0_seed120490.json 60 2

# 실험 1: 본 실행 (인스턴스 0-4)
for i in 0 1 2 3 4; do python exp1_classical_5x5/exact_only.py $i && python exp1_classical_5x5/safe.py $i; done
python exp1_classical_5x5/report.py

# 실험 2: 3x3 고전 (인스턴스 0-4, 약 2분)
for i in 0 1 2 3 4; do python exp2_classical_3x3/exact_only.py $i && python exp2_classical_3x3/safe.py $i; done
for i in 0 1 2 3 4; do python exp2_classical_3x3/exact_only_long.py $i; done   # safe.py 결과가 먼저 있어야 함 (약 3분)
python exp2_classical_3x3/report.py

# 실험 3: QPU 런 두 arm의 시작점 (인스턴스 0, c = 0.1, 수십 초, GPU)
python exp3_quantum_3x3/prepare_start_points.py
# 실험 3: QPU 학습 — exp3_quantum_3x3/train_qpu.ipynb를 그 폴더에서 열어 실행 (FakeNighthawk 기준 약 15분)
cd exp3_quantum_3x3 && jupyter nbconvert --to notebook --execute --inplace train_qpu.ipynb
```

완료된 결과 폴더는 덮어쓰지 않는다. smoke 결과(`exp1_smoke`)만 예외다.

## 지표 (`exp1_classical_5x5/report.py`, `exp2_classical_3x3/report.py`)

exact step은 t = 0..20이다. t = 0은 업데이트 전 시작점, t = 20은 20번 업데이트한 뒤다.
- `ar_exact_only_first/last`, `ar_safe_first/last`: 각 arm의 처음과 마지막 AR. SAFE의 first는 사전학습과 증류 직후다.
- `ar_safe_lwpp_only`: 증류 전 LWPP 최종 파라미터의 exact AR.
- `tau_target_*`: exact-only가 마지막에 도달한 AR에 처음 닿는 step. SAFE가 끝내 못 닿으면 비워 두고 실패로 센다.
- `tau099_*`: 자기 궤적 최고 AR의 99%에 처음 닿는 step (논문 정의).
- `circuits_per_step_*` = 2·N_active + 1: parameter-shift 기울기 한 번과 에너지 한 번에 드는 회로 수. 증류된 게이트는 회로에서 빠진다.
- `circuits_to_target_*` = τ × 스텝당 회로 수. `circuit_saving` = 1 − SAFE / exact-only.

연장 실행 지표 (`exp2_classical_3x3/report.py` → `summary_long.csv`). 목표는 각 (인스턴스, 초기값) 쌍에서 SAFE가 20 step 뒤 도달한 AR이다.
- `steps_safe_to_target`, `steps_exact_only_to_target`: 각 arm이 목표 AR에 처음 닿은 step. exact-only가 최대 step 안에 못 닿으면 비워 둔다.
- `steps_exact_only_to_safe_first`: exact-only가 SAFE의 시작 AR(사전학습·증류 직후)에 처음 닿은 step.
- `circuits_*_to_target` = step × (2·N_active + 1), `circuit_saving_to_target` = 1 − SAFE / exact-only. 실기기라면 여기에 회로당 shot 수가 곱해진다.
- `max_dE_first_steps_vs_exact_only`: 연장 실행의 앞부분이 `exact_only.py`와 같은지 확인하는 값 (0이어야 함).

## 저장 포맷 (실행 폴더마다)

- `problem.json`, `problem.png`: 푼 문제
- `circuit.json`: 게이트 목록(순서대로)과 각도 규약 R_P(θ) = exp(−iθP/2), 관측량
- `params.json`: 저장하는 값은 아래와 같다.
  - 초기값별 θ_init, (SAFE는) θ_lwpp_final과 남은 각도 번호(`active_params`)
  - 매 step의 θ, 에너지, AR, 그리고 θ_final
  - 설정, 패키지 버전, 시간과 메모리
- `qasm/*.qasm`: 초기·최종 파라미터를 대입하고 측정을 붙인 OpenQASM 2.0. 다른 하드웨어에서 그대로 실행할 수 있다.

참고: PennyLane의 torch 인터페이스는 결과를 torch 기본 dtype으로 돌려준다. 그래서 스크립트 첫머리에서 `torch.set_default_dtype(torch.float64)`로 설정한다. 이 설정이 없으면 exact 에너지가 float32로 반올림된다. PADO의 float32 계산 결과에는 영향이 없다.
