# CARE-O2O / ARW-O2O / RG-O2O

구현 근거는 사용자 제공 `3_cro2o_algorithms_for_test.pdf` (2026-09-15,
SHA256 `87f564451211923cfcbe80903d5066ec73b3e7617efaf365c7c4cdac20fbf62d`)이다.
문서의 세 이름은 **미출판 연구 후보**이며 기존 검증 baseline/main-table
allowlist에 추가하지 않았다. 구현 버전은 `cro2o_working_notes_20260915_v1`이다.

## 구현 범위

- 공통: 독립 critic/target ensemble, RIQL-style quantile/Huber/expectile/AWR
  초기화, tanh Gaussian 초기 actor, trajectory별 고정 Bernoulli bootstrap mask.
  동일 seed/config/data이면 세 후보의 초기 critic/value/actor 파라미터와
  RIQL 초기 학습 경로를 일치시킨다. 기존 RIQL 클래스와 bitwise parity를
  주장하지 않으며, 다른 알고리즘 명의 checkpoint 자동 변환은 제공하지 않는다.
- CARE-O2O: median/MAD value center와 epistemic dispersion, 불확실성으로
  조정한 residual trust, 가중 Huber critic, transition trust를 상속하지 않는
  SAC actor. 탐험 시에만 candidate utility에 optimism을 더한다.
- ARW-O2O: 같은 네트워크/신뢰도/수집 방식에 source-specific loss를 적용한다.
  두 minibatch의 손실을 **각각 정규화한 뒤** omega로 결합한다. distinct
  trajectory block의 suspicion/ESS와 residual MAD로 PDF 식 (3.2)를 계산한다.
  `--candidate-retention-mode adaptive|fixed|none`으로 동일한 source-normalized
  learner의 adaptive/fixed/no-retention 대조군을 실행할 수 있다.
- RG-O2O: 조건부 epsilon-prediction DDPM, weighted squared denoising,
  clipped advantage weights, delayed generator/critic의 exploitation backup.
  온라인 critic/정책 개선에는 log G, IPW, SAC entropy term을 사용하지 않는다.
  generator는 offline과 online 모두 실제 optimizer로 갱신된다.

PBRL two-view/OOD ablation, local state-validity estimator, 지역별 source rule,
모든 weighting ablation은 이 최소 구현에 추가하지 않았다.

## 데이터 정직성과 학습 순서

1. 오염된 원본 데이터와 trajectory ID로 고정 block을 만든다. baseline의
   clean shadow, corruption mask, `mc_calibration_valid` 등은 학습하지 않는다.
2. 공통 RIQL-style 초기화를 학습한다.
3. 각 audit fold마다 **처음부터 별도 네트워크**를 학습한다. held-out fold는
   critic/value/target, 학습용 normalizer와 residual scale 추정에서 제외한다.
   scale은 해당 training fold에서 optimizer step 전에 얻은 residual의
   running MAD로 추정하고, 그 fold의 held-out 점수 계산 중에는 고정한다.
   전체 데이터로 학습한 모델을 복사한 뒤 지연시키는 방식을 cross-fitting으로
   부르지 않는다. audit seed는 private CPU scope이며 learner CPU/MPS/CUDA
   RNG를 변경하지 않는다. 두 fold 이상을 위한 trajectory가 부족하면 실패한다.
4. RG는 이후 offline reliability/quality-weighted generator를 학습한다.
5. K0개 warm-up transition을 실제 corruption channel로 수집한다. 새 logged
   tuple은 optimizer가 보기 전에 점수화하고 initial trust를 저장한다.
6. 정확히 K0 수집 후 actor/generator를 갱신하지 않고 critic만 재보정한다.
   그 이후 optimistic 수집과 일반 온라인 학습을 시작한다.
7. replay row는 source, immutable initial trust, block identity를 sidecar로
   연결한다. bootstrap mask는 `(learner_seed, source, block)`으로 재구성되며
   replay sampling 때 새로 뽑지 않는다. ring-buffer overwrite는 새 tuple의
   metadata로 교체한다. replay gradient 횟수는 distinct evidence에 더하지 않는다.

ARW suspicion/ESS는 block 내 평균 weight/suspicion에 대해 계산한다.
offline/online residual scale과 block 통계는 분리된다. 작은 weight가 모두 같으면
ESS가 커지는 현상은 그대로 두며, suspicion 통계로 별도 구분한다.

## 명시적으로 선택한 기본값

PDF가 수치값을 고정하지 않은 항목의 구현 기본값이며 튜닝된 권장값이 아니다.

| 항목 | 기본값 |
|---|---:|
| critic 수 | 5 (`--num-critics`로 공통 조정) |
| fixed offline ratio | 0.5 |
| audit folds / fold별 gradient iterations | 2 / offline_steps |
| warm-up | max(initial_collection_steps, warmup_steps), 기본 5,000 |
| critic-only recalibration | 1,000 iterations |
| RG offline generator iterations | offline_steps |
| 후보 수 / checking probability | 10 / 0.05 |
| optimism / uncertainty allowance | 1 / 1 |
| MAD multiplier / scale floor / residual threshold | 1.4826 / 1 / 2 |
| trust minimum / temperature | 0.05 / 1 |
| bootstrap inclusion probability | 0.8 |
| residual scale window | 2,048 |
| ARW retention period / smoothing / bias scale | 100 collected transitions / 0.1 / 1 |
| advantage exponent clip L | 5 |
| diffusion steps / schedule | 20 / cosine cumulative alpha |

Huber delta는 기존 RIQL 설정의 `1 / riql_sigma**2`를 사용한다. AWR 상한은
`exp(L)`, RG advantage는 `[-L,L]`로 clip한다. CARE/ARW alpha는 1에서 시작해
target entropy `-action_dim`으로 학습한다. b_scale/scale floor는 reward/return의
현재 수치 단위이며, reward를 바꾸면 동일한 숫자가 동일 실험을 뜻하지 않는다.
Audit에는 fold별로 Q/value/actor optimizer가 있어 총 추가 optimizer 수는
`3 * candidate_audit_updates`이다. 이 비용과 generator/recalibration 비용은
completion metadata에 별도로 기록한다. 기본 audit 예산은 작지 않다.

## 기존 저장소와 PDF 사이의 명시적 차이

PDF 1쪽은 logged action도 action set 안이라고 가정하지만, 현재 benchmark는
범위 밖 action label을 허용하는 replay poisoning이다. 비교 조건을 바꾸지 않도록
**기존 corruption을 유지**했다. 실행 action과 생성한 diffusion action만 bounded이며
poisoned replay action을 새로 clip하지 않는다. 따라서 bounded-log 가정의 정확한
재현은 아니다. AWR likelihood는 기존 tanh Gaussian의 inverse-tanh clamp safeguard를
사용하며, critic/denoising에 들어가는 replay action 자체는 변경하지 않는다.

CARE/ARW 최종 평가는 optimism 없는 deterministic base actor다. RG는 PDF대로
optimism 없는 **확률적 diffusion candidate reranking**이다. argmax가 있다고
deterministic policy가 되지 않는다. RG 평가 행은 `generative_exploitation`,
manifest는 `stochastic_diffusion_exploitation_reranking`으로 기록하며
`return_deterministic` 열에 RG score를 채우지 않는다. 평가 전체에서 training RNG,
optimizer, audit state는 보존한다. RG의 candidate 수/생성 비용을 맞춘 대조군이 필요하다.

## 실행과 재개

저장소 루트에서 기존 `run_experiment.py`를 사용한다. 소문자 CLI 이름과 PDF 이름
`CARE-O2O`, `ARW-O2O`, `RG-O2O` alias 모두 허용한다.

```bash
python run_experiment.py --algorithm care_o2o --env-name hopper-medium-replay-v2
python run_experiment.py --algorithm arw_o2o --env-name hopper-medium-replay-v2 --candidate-retention-mode adaptive
python run_experiment.py --algorithm rg_o2o --env-name hopper-medium-replay-v2
```

`run_all_algorithms.py` / `run_matrix.py`에서는
`--algorithms care_o2o arw_o2o rg_o2o`를 명시한다. 새 후보는
`common_budget_robustness` 설정으로 실행하며, 기존 5개 전용 `research_benchmark`
또는 로컬 `run_55_experiment.py` allowlist에 강제로 넣지 않는다.

Checkpoint에는 model/optimizer와 함께 offline artifact identity, trust, block IDs,
online slot metadata, source residual history, distinct blocks, retention 및 모든
phase counter가 저장된다. 동일 실험의 episode-boundary 재개는 기존 `--resume-run`을
사용한다. 새 실험 초기화는 같은 후보의 offline checkpoint로 제한하며, online
checkpoint의 replay/audit state를 새 실험 초기값으로 오인해 재사용하지 않는다.

## 검증과 한계

`tests/test_cro2o.py`는 수식, 독립 critic, matched initializer, fold 분리,
private audit RNG, frozen warm-up/recalibration, replay-action 구분,
실제 backward/optimizer step, model/optimizer/RNG 재개, 평가 불변성을 검사한다.
실제 Hopper-v4와 실제 D4RL의 4개 trajectory(114 rows)로 세 후보 모두
`run_experiment`의 offline → audit → warm-up → recalibration → online → 평가
→ checkpoint 저장까지 축소 smoke를 수행했다. loader만 해당 subset으로 제한했다.
이는 장기 학습, 다중 seed 성능, 공격 탐지 정확도, robustness/novelty 증명이 아니다.
