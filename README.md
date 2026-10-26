# SAFE with PADO — TTA 시험 재현 레포

PADO-Pauli(`padopauli`)의 LWPP 대리 모델로 사전학습·증류한 뒤 exact 미세조정하는 학습 방식(PADO 방식)이, 처음부터 exact 학습만 하는 방식(standard 방식)보다 양자 자원을 얼마나 아끼는지 잰다. 시험 항목 TC1·TC2의 정의와 판정 기준은 시험합의서를 따르며, 같은 값이 `configs/TC*.json`에 들어 있다. 이 브랜치(`NPU`)에서는 TC1을 5x5 대신 4x4 격자로 돌리고(exact 평가를 CPU 상태벡터로 하므로), 양자컴퓨터 실행 시험(TC3)은 두지 않는다.

- PADO 방식 (`pado_safe`): LWPP 사전학습(NPU, 양자 자원 0) → |θ| ≤ 0.3인 게이트를 회로에서 제거(증류) → 남은 각도만 exact 미세조정
- standard 방식 (`standard`): 같은 초기값에서 exact 학습만. `standard_long`은 PADO 방식의 최종 AR에 닿을 때까지 연장한 실행
- exact 평가: PennyLane 상태벡터(CPU, `lightning.qubit`, adjoint 미분)
- 양자 자원: exact step T까지의 회로 수 C(T) = T × (2·N_act + 1) + 1. 고전 시뮬레이션이므로 parameter-shift로 기울기를 구했을 때의 환산값
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
   │ ╲   ╱ │ ╲   ╱ │       * 4x4 격자 (n=16): NN 24개 + NNN 18개 = 간선 42개, 필드 16개
   │   ╳   │   ╳   │
   │ ╱   ╲ │ ╱   ╲ │
  (6) ─── (7) ─── (8)
```

**회로 (ma-QAOA, 깊이 $p=3$)**: 초기 상태 $|+\rangle^{\otimes n} = H^{\otimes n}|0\rangle^{\otimes n}$ 위에, 층($l=1,\dots,p$)마다 **모든 게이트가 독립 파라미터 $\theta_k$를 갖는** 다중각도 QAOA 회로를 쌓는다 ($R_P(\theta) = \exp(-i\theta P / 2)$).

$$|\psi(\boldsymbol{\theta})\rangle = \left[ \prod_{l=1}^{p} \left( \prod_{i=0}^{n-1} R_X(\theta_{l,i}^X) \right) \left( \prod_{i=0}^{n-1} R_Z(\theta_{l,i}^Z) \right) \left( \prod_{(i,j) \in E} R_{ZZ}(\theta_{l,ij}^{ZZ}) \right) \right] H^{\otimes n} |0\rangle^{\otimes n}$$

- **층당 파라미터 수**: $|E|$개($R_{ZZ}$) + $n$개($R_Z$) + $n$개($R_X$) → **3x3($p=3$)은 총 114개**, **4x4($p=3$)는 총 222개**
- **PADO 증류**: LWPP 사전학습 후 $|\theta_k| \le 0.3$인 회전 게이트($R_{ZZ}, R_Z, R_X$)는 항등 연산($I$)에 가까우므로 회로에서 통째로 제거하고, 살아남은 $N_{\text{act}}$개 게이트만으로 미세조정한다.

| | TC1 | TC2 |
|---|---|---|
| 문제 | 4x4 격자(16큐비트), 인스턴스 5개 | 3x3 격자(9큐비트), 인스턴스 5개 |
| 초기값 | 상수 5종 (RZZ·RZ = +c, RX = −c, c = 0.01, 0.05, 0.1, 0.2, 0.4) | 같음 |
| LWPP 사전학습 | max_weight 5, 200 step | max_weight 3, 100 step |
| exact 단계 | PennyLane, 20 step. standard는 PADO 최종 AR까지 연장(상한 500) | PennyLane, 20 step. 연장 상한 2000 |
| 실행 환경 | Tenstorrent P300C(LWPP 사전학습) + CPU(exact) | 같음 |

## 실행 (시험자용)

환경을 활성화하고 Pado-Lab-TT의 `env.sh`를 source 한 셸에서 시험 항목마다 한 줄이다. `env.sh`가 현재 폴더를 tt-metal로 옮기므로(NPU 커널 소스가 tt-metal 안의 상대경로) 스크립트는 레포 경로를 붙여 부른다. 결과는 `results_on_P300C/<TC>/`에, 지표·판정·그림은 그 안의 `metrics/`에 쓴다.

```bash
conda activate <env>                                       # padopauli와 padopauli-tt가 설치된 환경 활성화
source <Pado-Lab-TT>/env.sh                                # TT_METAL_HOME, PYTHONPATH, 현재 폴더 = tt-metal
python <repo>/run.py --exp TC1 && python <repo>/metric.py --exp TC1      # 4x4 고전
python <repo>/run.py --exp TC2 && python <repo>/metric.py --exp TC2      # 3x3 고전
python <repo>/run.py --exp TC1_smoke && python <repo>/metric.py --exp TC1_smoke   # 코드 점검용 축소판 (3x3)
```

보드는 한 번에 한 작업만 쓴다. 다른 작업이 보드를 쓰는 동안에는 실행하지 않는다.

`run.py`는 단계마다 `===== [TC1] ...` 줄을 찍는다: 환경 기록 → 문제 파일 확인(생성 스크립트를 다시 돌려 전수 조사 결과가 저장된 파일과 같은지 검사) → PADO-Pauli와 PennyLane의 에너지·기울기 일치 검사(호스트 엔진 float64로 판정하고, 같은 회로의 NPU float32 차이는 출력만 한다) → 인스턴스마다 `standard` → `pado_safe` → `standard_long`. 각 스크립트도 자기 단계(`1/3 LWPP 사전학습`, `2/3 증류`, `3/3 exact 미세조정`)와 step별 AR을 찍는다. 끝난 실행은 건너뛰므로 중단된 뒤 다시 실행하면 이어서 돈다. 완료된 결과 폴더는 덮어쓰지 않는다(`*_smoke`만 예외). 연장 실행은 config의 `run_standard_long`(true/false)으로 켜고 끄며, 상한은 `standard_long_max_steps`다(시험합의서대로 TC1 500, TC2 2000). 연장 실행을 끄면 도달 쌍 수와 절감률 R은 계산되지 않는다.

`metric.py`는 `configs/<TC>.json`의 `criteria`로 판정하고 기준마다 PASS/FAIL과 값을 출력한다.

## 환경 (Tenstorrent NPU)

- PADO-Pauli 엔진: PADO-Lab 브랜치 `workstation_for_npu`의 `padopauli` 2.1.0, editable `--no-deps` 설치 (ttnn이 numpy 2 미만을 요구한다)
- NPU 경로: Pado-Lab-TT 브랜치 `HW`의 `padopauli-tt`, editable `--no-deps` 설치. `compile_program(..., preset="tenstorrent")`가 entry point로 이 포트에 가고, 계산은 float32만 지원한다
- ttnn: 포트의 커스텀 op를 넣어 빌드한 tt-metal. 설치 절차는 Pado-Lab-TT README
- exact 평가: `pennylane`, `pennylane-lightning`(`lightning.qubit`)

## 구성

```
configs/                    시험 설정 (문제·회로·학습·판정 기준): TC1.json, TC2.json, TC1_smoke.json
run.py                      시험 실행기        python run.py --exp TC1
metric.py                   지표·판정·그림     python metric.py --exp TC1
problems/                   문제 생성 스크립트(make_problems.py), 인스턴스 JSON(전수 조사 해 포함), PNG
classical/                  TC1·TC2: standard.py, pado_safe.py, standard_long.py, check_pado_vs_pennylane.py, common.py
results_on_P300C/<TC>/      결과 (git에 넣는다)
```

## 결과 파일

- `results_on_P300C/TC1/<문제>/{standard,pado_safe,standard_long}/`
  - `params.json`: 설정, 패키지 버전, 초기값별 θ_init·θ_final, step별 θ·에너지·AR·시간. `pado_safe`는 LWPP compile·학습 시간, 호스트 peak RSS, NPU DRAM 할당량(단계 끝에 읽은 값), 증류 뒤 남은 각도 번호(`active_params`), 증류 전 exact 에너지도. `standard_long`은 목표 AR과 도달 step.
  - `circuit.json`(게이트 목록, 각도 규약 R_P(θ) = exp(−iθP/2)).
  - 문제 파일 `problem.json/png`는 실행 폴더가 아니라 문제 폴더(`results_on_P300C/TC1/<문제>/`)에 하나만 둔다.
- `results_on_P300C/TC1/env.txt`: 패키지 목록, 엔진(PADO-Lab)과 포트(Pado-Lab-TT)의 커밋, 칩마다 보드 이름.
- `metrics/` (metric.py): `pairs.csv`(쌍별 AR·τ·C·R·D·환산 shots), `criteria.json`(값, 임계값, PASS/FAIL, 전체 판정, 실행 상태), `energy_curves*.png`.

## 지표와 판정 기준 (시험합의서 2.4 라, 시험규격서)

`metric.py`가 `configs/<TC>.json`의 산정식·`criteria`대로 계산해 `metrics/pairs.csv`(쌍별), `metrics/criteria.json`(판정)에 쓴다. 모든 AR은 exact 평가로 계산하며 LWPP 대리 모델의 에너지는 지표에 쓰지 않는다.

### 측정 지표 (TC1·TC2, 쌍마다)

| 지표 | 정의 | 의미 | `pairs.csv` 열 |
|---|---|---|---|
| AR(T) | (E_max − E)/(E_max − E_min). E는 exact step T의 에너지(T = 0은 시작점), E_min·E_max는 전수 조사 | 해의 품질 (1 = 최적) | `AR_std_0`, `AR_std_20`, `AR_pado_0`, `AR_pado_20` |
| N, N_act | 전체 각도 수(4x4: 222, 3x3: 114), 증류 후 남은 각도 수 | 회로 크기 | `N`, `N_act` |
| D | N_act / N | 잔존 비율 (step당 회로 수 비율) | `D` |
| ΔAR(20) | AR_PADO(20) − AR_std(20) | 같은 exact step 예산에서의 품질 이득 | `delta_AR20` |
| A*, A_0 | A* = AR_PADO(20), A_0 = AR_PADO(0) (쌍마다 정함) | 목표 품질, 시작점 품질 | = `AR_pado_20`, `AR_pado_0` |
| τ(A) | AR ≥ A에 처음 도달하는 exact step 수. standard는 상한 T_max(TC1 500, TC2 2000) 안에 못 닿으면 미도달 | 목표 품질에 드는 비용 | `tau_pado_Astar`, `tau_std_Astar`, `tau_std_A0`, `reached` |
| C(T), S(T) | C(T) = T × (2·N_act + 1) + 1 (standard는 2·N + 1), S(T) = C(T) × 1024 | step T까지의 누적 회로 수, 환산 shots | `C_pado`, `C_std`, `shots_pado_1024`, `shots_std_1024` |
| R | 1 − C_PADO(τ_PADO(A*)) / C_std(τ_std(A*)). 미도달 쌍은 제외 | 같은 품질까지의 회로 절감률 | `R` |
| 고전 비용 | LWPP compile 시간, 사전학습 시간, 호스트 peak RSS, NPU DRAM 할당량 | 절감의 대가로 쓴 고전 자원 (기록) | `lwpp_compile_s`, `lwpp_train_s`, `lwpp_peak_rss_gb`, `lwpp_dram_gb` |
| Status | 실행의 정상 완료 여부 | 실행 가능성 | `status_standard`, `status_pado_safe`, `status_standard_long` |

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

`criteria` 키 끝의 `_gt`/`_ge`/`_le`가 비교 방향이고, `metric.py`는 기준마다 값·임계값·PASS/FAIL을 출력한다. TC마다 모든 기준을 만족하면 전체 PASS.

참고: PennyLane의 torch 인터페이스는 결과를 torch 기본 dtype으로 돌려주므로 스크립트 첫머리에서 `torch.set_default_dtype(torch.float64)`로 설정한다. 이 설정이 없으면 exact 에너지가 float32로 반올림된다.
