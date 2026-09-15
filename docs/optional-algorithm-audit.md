# 추가 알고리즘 구현 검토

검토 기준: `44cc45a` (five-baseline benchmark 수정본). 이 문서는 PEX,
RIQL+PEX, UWMSG, RO2O의 코드 검토와 CPU 고정 배치 검사 결과이다.
실환경 성능, 논문 점수, 전체 optimizer trajectory 일치를 인증하지 않는다.
이번 검토에서는 추가 알고리즘의 학습 코드를 변경하지 않았다.

## 버전과 복구

- `codex/before-benchmark-reliability`: 수정 전 커밋
  `e3afb9bba57e9b19f8261c74038df346a327e827`.
- `codex/benchmark-reliability`: 수정본과 이 검토 기록/테스트.
- `44cc45a`: 기존 Documents 수정본의 첫 커밋.
- `comparison.ipynb`의 사용자 변경은 커밋에 포함하지 않았다. 각 로컬
  폴더의 노트북은 내용이 서로 다르며 각각 보존한다.

저장소 안에서 `git switch codex/before-benchmark-reliability`로 수정 전
코드로 이동하고, `git switch codex/benchmark-reliability`로 돌아온다.
브랜치 전환은 기존 results/cache/checkpoint 파일의 백업·복원이 아니다.
구조가 변경된 PQE checkpoint를 다른 구조의 코드로 그대로 재사용할 수 없다.

## 참조한 공식 소스

| 대상 | 확인한 커밋 | 검토 경로 |
|---|---|---|
| PEX | [b0ae11e17e5fd9ae38840748ce043aff895e952b](https://github.com/Haichao-Zhang/PEX/tree/b0ae11e17e5fd9ae38840748ce043aff895e952b) | `pex/algorithms/{iql,pex}.py`, `pex/networks/policy.py`, `main_online.py` |
| RIQL+PEX | [35da71ee5151b6179d21b9a2b4ce1b6408aedd04](https://github.com/felix-thu/RPEX/tree/35da71ee5151b6179d21b9a2b4ce1b6408aedd04) | `pex/algorithms/riql_pex.py`, `riql.py` |
| UWMSG | [8d44a3efe1972c318b5ff5fb4b7fc2810e651931](https://github.com/YangRui2015/UWMSG/tree/8d44a3efe1972c318b5ff5fb4b7fc2810e651931) | `UWMSG.py`, `configs.py` |
| RO2O | [d066702273944fa5ec15a3050c05cf30f1ce9e05](https://github.com/BattleWen/RO2O/tree/d066702273944fa5ec15a3050c05cf30f1ce9e05) | `algorithms/offline/ro2o_mujoco.py`, `algorithms/finetune/ro2o_ft_mujoco.py` |

## 결론

| 알고리즘 | 확인한 구현 | 남은 판단 |
|---|---|---|
| RIQL+PEX | ensemble quantile, Huber, expectile 경로; frozen offline actor + 새 online actor; IPW 없는 Q gate; actor/critic/value 실제 갱신 | 핵심 구조는 검토 범위에서 일치. 기본 deterministic 평가는 upstream epsilon 평가와 다르다. 전체 수치 parity 및 실환경 검증은 미완료 |
| PEX | twin-Q IQL, expectile/AWR, frozen actor와 Q-only expansion, offline/online replay 혼합 | 기본 actor 분포·깊이 및 optimizer phase 전환이 공식 PEX와 다르므로 현재 설정은 변형 구현 |
| UWMSG | Q mean minus LCB·std actor objective, head별 Bellman target, detached uncertainty-weighted TD loss | 공식 핵심 TD 수식은 일치하지만 default table 오류, 명시적 override 덮어쓰기, actor 구조 차이 존재. 온라인 사용은 offline 알고리즘의 custom extension |
| RO2O | offline policy KL/Q smoothness/OOD penalty의 gradient; min-Q online 업데이트 | TD reduction 배율, actor 구조, perturbation sampling, offline resume 상태 문제 존재. 논문용 원본 구현으로 사용하기 전에 정리 필요 |

현재 `research_benchmark`가 허용하는 이름은 기존 5개 baseline이다.
추가 4개는 CLI에서 실행 가능하다는 이유만으로 검증된 main baseline에
자동 편입되지 않는다. RIQL+PEX는 source commit이 있지만 main registry에는
없고, PEX/UWMSG/RO2O도 이 검토만으로 등록·인증하지 않았다.

## 우선 수정할 항목

### P1 — UWMSG table 오류와 명시적 override 덮어쓰기

`robust_o2o/agents/registry.py::_apply_uwmsg_defaults`는 값이 `4.0` 또는
`0.7`이면 사용자 지정인지 구분하지 않고 변경한다. 실제 production
constructor probe에서 Walker2d/random/rewards에 지정한
`lcb_ratio=4.0, uncertainty_ratio=0.7`이 agent 생성 후 `6.0, 0.3`으로
변경되었다.

공식 `configs.py`는 Walker2d/random/**rewards**의 LCB를 `4.0`,
**dynamics**의 LCB를 `6.0`으로 구분한다. 현재 port는 두 경우 모두
`6.0`으로 만든다. 미지정 값과 명시적 값을 구분해 config resolver에서
한 번만 결정하고, 해당 target별 table을 수정해야 한다.

### P1 — RO2O critic TD loss가 공식 코드 대비 1/N

공식 offline/finetune 코드는 critic별 batch mean을 구한 뒤 critic들을
**sum**한다. 현재 `SACEnsembleAgent.update`는 critic과 batch 전체를
**mean**한다. Offline smoothness/OOD 항은 같은 방식으로 나누지 않으므로
단순 표시 차이가 아니라 정규화 항 대비 TD loss 비율을 바꾼다.

공식 `_critic_loss` 함수 자체를 AST로 추출해 동일 네트워크/terminal
고정 배치에 적용했다. 이 probe는 target randomness를 제거하고 RO2O의
정규화 항을 0으로 하여 TD reduction만 비교했다. 결과:

| 방법 | 공식 함수 loss | 현재 production update의 loss | 공식/현재 |
|---|---:|---:|---:|
| UWMSG, N=10 | 3.4846143723 | 3.4846143723 | 1.0 |
| RO2O, N=10 | 3.4846143723 | 0.3484615088 | 약 10.0 |

UWMSG의 전체 stochastic optimizer parity를 확인했다는 뜻은 아니다.
이 차이를 의도적인 정규화로 유지하려면 regularizer 계수와 함께 명시한
변형으로 취급해야 하며, 단순 critic 개수 통일과 구분해야 한다.

### P1 — RO2O offline resume 시 uncertainty schedule 소실

`self.ro2o_uncertainty`는 `_ro2o_ood_penalty`에서 감소하는 Python scalar이다.
`BaseAgent.checkpoint_state`에는 model/optimizer/counter만 저장되며 이 값은
포함되지 않는다. 실제 저장/복원 probe에서 `0.9999995`가 `1.0`으로
되돌아갔다. Offline 학습을 재개하면 OOD penalty의 계수가 달라진다.
상태를 checkpoint에 저장하고 연속 실행과 resume의 다음 update를 비교해야 한다.

### P2 — 추가 알고리즘 actor 구조가 공식 코드와 다름

`TanhGaussianPolicy`의 generic trunk는 `mlp(..., output_dim=hidden_dim)`을
사용하므로 `hidden_layers` 뒤에 affine 하나가 추가된다.

- PEX: 기본 `hidden_layers=2`인데 actor trunk는 affine 3개. 공식 PEX는
  2개 ReLU hidden block과 별도 mean head를 사용한다. 현재 기본은
  tanh-squashed/state-dependent-std이고, 공식 launcher는
  bounded-mean unsquashed/state-independent-std Gaussian이다.
- UWMSG/RO2O: `max(hidden_layers, 3)`에 generic output affine이 추가되어
  actor trunk가 affine 4개. 공식 actor는 3개 ReLU hidden block이다.
  Hidden bias 초기화도 다르다.

정책 변경은 likelihood, entropy, corruption label 처리에 영향을 준다.
공식 구조를 복구할지 공통 구조로 재설계할지는 실험 목적에 따라 정하되,
그 결정을 별도의 구조/정책 설정으로 기록해야 한다.

### P2 — RO2O perturbation sampling의 차이

공식 `get_noised_obs`는 `[sample_size, state_dim]` perturbation을 한 번
뽑아 batch의 모든 state에 공유한다. 현재 `_ro2o_noised_states`는
`[batch_size, sample_size, state_dim]`을 뽑아 각 state에 독립적으로 적용한다.
개별 state의 marginal noise 범위는 같지만 batch 내 상관과 RNG 소비가
달라진다. 이 선택을 문서화하고 fixed-noise 목적함수 검증이 필요하다.

## 오인하면 안 되는 source choice

RO2O 논문은 학습 objective의 연속성을 강조하지만, 확인한 공식 MuJoCo
finetune 파일은 smoothness/OOD 항을 **계산만 하고 최종 loss 합산에서
제외**한다 (`loss = q_loss`, `loss = critic_loss1`). 현재 port가 online에서
이 항들을 끄는 것 자체를 확정 버그로 판정하지 않았다. 다만 항 계산 여부에
따른 RNG 소비는 source와 다르다. 원 논문 수식과 공개 코드 중 어떤 경로를
baseline으로 삼을지 별도로 명시해야 한다.

PEX는 online용 actor optimizer를 새로 만들지만 기존 critic/value optimizer
state를 유지한다. 공식 `main_online.py`의 새 process/optimizer 생성과는
차이가 있다. RIQL+PEX의 deterministic evaluation 역시 공통 평가 adaptation이며
upstream epsilon-switching 평가의 exact reproduction은 아니다.

## 검증

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_optional_algorithm_contracts.py
```

새 파일의 8개 검사는 각각 실제 production agent/function을 호출한다:

- 추가 4개 방법별 offline 2 update + online 2 update, finite metrics/parameters,
  actor/critic/value 실제 변경, target 변경 및 target gradient 없음.
- PEX/RIQL+PEX의 frozen actor 불변성과 새 actor parameter 독립성.
- PEX/RIQL+PEX의 Q-only gate 및 deterministic tie-break.
- UWMSG LCB value/gradient와 RO2O offline regularizer들의 gradient.
- Deterministic action의 finite/bounds 및 learner Torch RNG 불변성.

전체 결과: **268 passed, 58 subtests passed, 3 warnings**.
경고는 cache multiprocessing의 fork deprecation 2개와 legacy checkpoint
provenance 경고를 의도적으로 발생시키는 검사 1개이다.
실환경 학습/성능 sweep은 실행하지 않았다. 통과하는 실행 테스트가 위의
source 차이 또는 resume 문제를 해소하지는 않는다.

## 이전 검토에서 확인한 공통 미해결 항목

이 브랜치는 현재 수정본의 보존 지점이며 다음 문제를 해결한 버전은 아니다:

1. Offline adversarial cache 생성과 cache hit에서 공유 oracle RNG 상태가
   달라져 같은 seed/입력의 online 공격 결과가 달라질 수 있음.
2. Seed별 actual-change 통계와 artifact cache key/path가 aggregation
   signature에 남아 동일 설정의 seed 집계를 거부할 수 있음.
3. `evaluation_seed_role=tuning/final`은 용도 구분이며 자동 disjoint seed
   분리가 아님. 현재는 별도 `eval_seed`를 명시해야 함.
4. 제공된 EDAC checkpoint preprocessing은 metadata 부족으로 미검증.
   Documents 복사본에서는 sibling `RIQL-main`이 없으면 default checkpoint를
   찾지 못하므로 명시적 checkpoint 경로/hash가 필요함.

Critic 수·UTD 통일안은 아직 적용하지 않았다. 현재 snapshot의 research WSRL은
10 critics/UTD 4이고, 각 방법의 정책 분포도 기존 수정본 그대로이다.
