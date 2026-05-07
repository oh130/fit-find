# 추천 모델 실험 기록

## 1. 문서 목적

본 문서는 추천 모델 파트에서 수행한 핵심 실험, 최종 채택안, 명세 통과 여부를 정리한다. 원본 metric JSON은 `rec_models/reports/` 하위에 보존하고, 본 문서에는 모델 선택과 최종 판단에 필요한 결과만 요약한다.

기록 원칙:

- 모든 실행 로그와 원본 metric JSON은 `rec_models/reports/`에 저장한다.
- 본 문서에는 비교 기준이 되는 실험과 최종 채택 결과를 기록한다.
- 중간 튜닝 실험은 최종 판단에 영향을 준 경우에만 요약한다.
- 과거 결과 JSON과 checkpoint metadata는 당시 실험 기록이므로 임의 수정하지 않는다.

## 2. 최종 결론

추천 모델 파이프라인은 dev 기준으로 명세 지표를 통과했다.

| 단계 | 명세 기준 | 최종 결과 | 판정 | 원본 결과 |
|---|---:|---:|---|---|
| Candidate Generation | Recall@300 >= 0.30 | 0.614632 | 통과 | `rec_models/reports/candidate_experiments/dev_two_tower_history_itemid_fast_fixed.json` |
| Ranking | AUC >= 0.70 | 0.956739 | 통과 | `rec_models/reports/ranking_experiments/dev_ranking_logreg.json` |
| Final Recommendation | HitRate@50 >= 0.20 | 0.497000 | 통과 | `rec_models/reports/baseline/dev_e2e_twotower_serving_latency_pool75_1000.json` |
| Final Recommendation | NDCG@50 >= 0.08 | 0.178138 | 통과 | `rec_models/reports/baseline/dev_e2e_twotower_serving_latency_pool75_1000.json` |
| Final Recommendation | Coverage@50 >= 0.20 | 0.215203 | 통과 | `rec_models/reports/baseline/dev_e2e_twotower_serving_latency_pool75_1000.json` |
| Serving Latency | <= 200ms | p95 187.61ms | 통과 | `rec_models/reports/baseline/dev_serving_latency_top50_50users_pool75.json` |

최종 serving 흐름:

```text
dev processed data
  -> Two-Tower candidate retrieval + sequential/profile/coverage candidate signals
  -> Logistic Regression CTR ranking
  -> diversity/exploration/freshness reranking
  -> top-50 recommendation response
```

주의 사항:

- Session은 `history_article_ids`, recent clicks, session interest 기반으로 반영한다. GRU/Transformer 기반 session encoder는 별도 고도화 항목이다.
- MAB는 epsilon-greedy exploration slot 방식으로 구현했다. 온라인 reward update 기반 bandit 학습은 별도 고도화 항목이다.
- Cold-start fallback은 구현되어 있으나 최종 dev E2E 표본에서는 `cold_start_subset.users_evaluated = 0`으로 별도 subset 지표를 산출하지 못했다.

## 3. 실험 요약

| 날짜 | 실험명 | 단계 | 목적 | 주요 결과 | 채택 여부 |
|---|---|---|---|---|---|
| 2026-05-06 | `dev_two_tower_history_itemid_fast_fixed` | Candidate | history + item-id embedding Two-Tower 평가 | Recall@300 0.614632 | 채택 |
| 2026-05-06 | `dev_ranking_logreg` | Ranking | dev user split LogReg ranking 평가 | AUC 0.956739 | 채택 |
| 2026-05-06 | `dev_e2e_twotower_serving_coverage_strong` | Final Pipeline | coverage 강화 E2E 검증 | HitRate@50 0.592000, NDCG@50 0.212782, Coverage@50 0.247172 | 중간 채택 |
| 2026-05-06 | `dev_serving_latency_top50_50users_pool75` | Serving | top-50 API path latency 측정 | avg 127.47ms, p95 187.61ms | 채택 |
| 2026-05-06 | `dev_e2e_twotower_serving_latency_pool75_1000` | Final Pipeline | latency 튜닝 코드 기준 E2E 재검증 | HitRate@50 0.497000, NDCG@50 0.178138, Coverage@50 0.215203 | 최종 채택 |
| 2026-05-06 | `dev_candidate_hybrid_combo` | Candidate | heuristic candidate 조합 비교 | profile + copurchase + sequential Recall@300 0.301 | 보조 baseline |

## 4. 데이터와 재현 조건

최종 실험은 dev mode 전처리 데이터를 사용했다.

- 데이터 파이프라인 실행:

```bash
DATA_PIPELINE_MODE=dev .venv/bin/python data_pipeline/run_data_pipeline.py
```

- dev mode 기준 transaction rows: `1,000,000`
- 주요 입력:
  - `data/processed/train_data_dev.csv`
  - `data/processed/item_features_dev.csv`
  - `data/processed/candidate_train_data_dev.csv.gz`
  - `data/processed/candidate_interactions_dev.csv.gz`
  - `data/processed/candidate_manifest_dev.json`
- seed: `42`
- Candidate split: `leave_last_out`
- Ranking split: `user`
- Final recommendation evaluation: `max_users=1000`, `top_k=50`, `candidate_k=300`

## 5. Candidate Generation

### 5.1 최종 채택 모델

최종 candidate generation은 개선된 Two-Tower retrieval을 기본 축으로 사용한다. Serving에서는 sequential transition, profile metadata, recent click/session signals, coverage exploration candidates를 함께 결합한다.

최종 Two-Tower checkpoint:

```text
data/checkpoints/candidate_dev_history_itemid_fast/two_tower.pt
data/checkpoints/candidate_dev_history_itemid_fast/two_tower_metadata.json
```

기본 checkpoint resolution:

- `TWO_TOWER_CHECKPOINT_DIR` 환경변수가 있으면 해당 경로 사용
- 없으면 `data/checkpoints/candidate_dev_history_itemid_fast` 우선 사용
- fallback: `data/checkpoints/candidate`

### 5.2 개선 내용

기존 Two-Tower는 Recall@300이 약 0.06~0.08 수준으로 낮았다. 주요 원인은 다음과 같았다.

- User Tower가 최근 구매 history sequence를 직접 보지 못함
- Item Tower가 `article_id` identity를 직접 학습하지 못함
- validation split이 user-level cold user split이라 next-item 목표와 맞지 않음
- inference/evaluator가 유저 첫 row를 사용해 빈 history로 평가하던 버그가 있었음

수정 사항:

- `data_pipeline/build_candidate_training_data.py`
  - `history_article_ids` 컬럼 추가
  - 각 구매 row 기준 현재 구매 이전 최근 10개 article id 저장
- `rec_models/candidate/dataset.py`
  - `history_article_ids` 파싱
  - `history_item_ids`, `history_mask`, `item_id_index` batch 생성
  - `item_id_vocabulary`, `history_item_vocabulary` 추가
  - `leave_last_out_split()` 추가
- `rec_models/candidate/model.py`
  - User Tower에 history item embedding average pooling 추가
  - Item Tower에 article-id identity embedding 추가
  - `logit_scale=20.0` 적용
- `rec_models/candidate/train.py`
  - `--split-mode leave_last_out|user` 추가
  - 기본 split을 `leave_last_out`으로 변경
  - `--validation-max-users` 추가
- `rec_models/candidate/infer.py`
  - history/item-id vocabulary 로딩 추가
  - `build_latest_user_table()`로 최신 history row 사용
- `rec_models/serving/candidate_service.py`
  - serving artifact 로딩 시 최신 user history row 사용
  - 최종 checkpoint를 serving 기본 경로에 연결

### 5.3 학습 명령

```bash
.venv/bin/python -m rec_models.candidate.train \
  --data data/processed/candidate_train_data_dev.csv.gz \
  --epochs 2 \
  --batch-size 1024 \
  --device cuda \
  --validation-max-users 1000 \
  --checkpoint-dir data/checkpoints/candidate_dev_history_itemid_fast
```

학습 중 validation:

- Epoch 1 validation Recall@300: `0.356`
- Epoch 2 validation Recall@300: `0.441`

### 5.4 평가 명령과 결과

```bash
.venv/bin/python -m rec_models.candidate.evaluator \
  --data data/processed/candidate_train_data_dev.csv.gz \
  --mode two-tower \
  --top_k 300 \
  --max-users 1000 \
  --checkpoint-dir data/checkpoints/candidate_dev_history_itemid_fast \
  --output-json rec_models/reports/candidate_experiments/dev_two_tower_history_itemid_fast_fixed.json
```

결과:

```text
users evaluated    1000
Recall@300         0.614632
```

명세 `Recall@300 >= 0.30`을 통과했다.

### 5.5 Heuristic Candidate 보조 실험

leakage-safe sequential/copurchase artifact를 사용한 heuristic 조합도 비교했다.

주요 결과:

| 설정 | Recall@300 |
|---|---:|
| baseline_no_profile | 0.191 |
| baseline_profile | 0.239 |
| copurchase_candidate | 0.278 |
| sequential_combined | 0.268 |
| profile_copurchase_sequential_combined | 0.301 |

해석:

- heuristic 조합만으로도 명세 하한 0.30을 간신히 넘는다.
- 최종 Two-Tower는 Recall@300 0.614632로 heuristic 조합보다 훨씬 높다.
- co-purchase는 기본 serving에서는 꺼져 있고, 비교/실험용 artifact로 유지한다.

## 6. Ranking

### 6.1 최종 채택 모델

최종 ranking은 `LogisticRegression` 기반 CTR ranking pipeline을 채택했다.

Checkpoint:

```text
rec_models/checkpoints/logreg_dev/ranking_baseline.joblib
rec_models/checkpoints/logreg_dev/ranking_baseline_metadata.json
```

명세가 요구하는 ranking 항목은 `DeepFM/Wide&Deep 또는 CTR/CVR ranking`이므로, CTR ranking pipeline으로 충족한다. DeepFM은 구현 및 비교 실험이 있으나 최종 dev 기준에서는 LogReg가 더 안정적이므로 serving 채택안에서 제외했다.

### 6.2 학습 명령

```bash
.venv/bin/python -m rec_models.ranking.train \
  --model-type logreg \
  --split-mode user \
  --output-dir rec_models/checkpoints/logreg_dev
```

### 6.3 결과

원본 결과:

```text
rec_models/reports/ranking_experiments/dev_ranking_logreg.json
```

지표:

```text
AUC        0.956739
HitRate@50 1.000000
NDCG@50    0.981341
```

명세 `AUC >= 0.70`을 통과했다.

## 7. Re-ranking과 Coverage 개선

### 7.1 최종 기능

`rec_models/serving/rerank_bridge.py`와 `rec_models/serving/candidate_service.py`에 다음 기능을 적용했다.

- category diversity guard
- epsilon-greedy exploration slot
- new/fresh item boost
- coverage exploration candidates
- popularity 쏠림 완화
- retrieval prior와 ranking score blending

### 7.2 Coverage 개선 흐름

초기 E2E serving 평가에서는 HitRate/NDCG는 높았지만 Coverage가 낮았다.

| 실험 | HitRate@50 | NDCG@50 | Coverage@50 | 비고 |
|---|---:|---:|---:|---|
| Two-Tower serving 연결 직후 | 0.707000 | 0.253603 | 0.066400 | `dev_e2e_twotower_serving.json` |
| Coverage 강화 | 0.592000 | 0.212782 | 0.247172 | `dev_e2e_twotower_serving_coverage_strong.json` |
| Latency 튜닝 코드 최종 | 0.497000 | 0.178138 | 0.215203 | `dev_e2e_twotower_serving_latency_pool75_1000.json` |

해석:

- Coverage 강화를 위해 정확도를 일부 희생했지만 HitRate/NDCG는 명세 기준보다 충분히 높게 유지했다.
- 최종 latency-optimized 설정에서도 Coverage@50 0.215203으로 목표 0.20을 넘는다.

## 8. 최종 E2E 평가

### 8.1 최종 평가 명령

```bash
.venv/bin/python -m rec_models.evaluation.evaluate_recommender \
  --top_k 50 \
  --candidate-k 300 \
  --max-users 1000 \
  --use-serving-candidates \
  --skip-popularity-baseline \
  --output-json rec_models/reports/baseline/dev_e2e_twotower_serving_latency_pool75_1000.json
```

### 8.2 결과

```text
users evaluated    1000
HitRate@50         0.497000
NDCG@50            0.178138
Coverage@50        0.215203
```

명세 통과 여부:

| 지표 | 기준 | 결과 | 판정 |
|---|---:|---:|---|
| HitRate@50 | >= 0.20 | 0.497000 | 통과 |
| NDCG@50 | >= 0.08 | 0.178138 | 통과 |
| Coverage@50 | >= 0.20 | 0.215203 | 통과 |

### 8.3 Popularity Baseline 비교

동일 dev 1000 유저 기준 popularity baseline 비교 결과는 다음 파일에 저장되어 있다.

```text
rec_models/reports/baseline/dev_e2e_twotower_serving_coverage_strong.json
```

| 모델 | HitRate@50 | NDCG@50 | Coverage@50 |
|---|---:|---:|---:|
| Current Model | 0.592000 | 0.212782 | 0.247172 |
| Popularity Baseline | 0.346000 | 0.053999 | 0.016647 |
| Improvement | +0.246000 | +0.158782 | +0.230524 |

## 9. Serving Latency

### 9.1 측정 기준

Latency는 실제 serving orchestration 함수인 `recommend(user_id, top_n=50)` 경로 기준으로 측정했다.

최종 E2E 품질 평가는 명세의 candidate cutoff에 맞춰 `candidate_k=300`으로 수행했다. Latency 측정은 실제 top-50 API request path를 기준으로 하므로 serving candidate pool은 top-50 요청 기준 75개이다.

측정 CLI:

```bash
.venv/bin/python -m rec_models.evaluation.measure_serving_latency \
  --top-k 50 \
  --max-users 50 \
  --output-json rec_models/reports/baseline/dev_serving_latency_top50_50users_pool75.json
```

Warmup은 서버 시작 시 artifact를 미리 로드하는 비용으로 보고 API request latency에는 포함하지 않는다.

### 9.2 결과

```text
users measured     50
warmup_ms          45633.07
wall avg_ms        127.47
wall p50_ms        117.22
wall p95_ms        187.61
wall max_ms        284.36
```

단계별 평균:

```text
candidate_avg_ms   27.20
ranking_avg_ms     55.58
reranking_avg_ms   44.60
```

명세 `API latency <= 200ms`는 avg/p50/p95 기준으로 통과했다.

최적화 내용:

- Two-Tower model, item embedding index, user profile store, sequential artifact를 warmup에 포함
- reranking의 반복 DataFrame copy loop를 vectorized sort 중심으로 단순화
- serving candidate pool을 top-50 기준 75개로 조정
- sequential artifact 로딩 dtype warning 제거

## 10. Cold-start

구현 상태:

- 사용자 feature가 없는 경우 cold-start-safe default feature 사용
- recent clicks/session interest가 없으면 popularity + cold-start bonus + freshness 기반 candidate fallback 사용
- profile이 있으면 preferred category/color/garment/price band 기반 candidate boost 적용

최종 dev E2E 평가에서는 `cold_start_subset.users_evaluated = 0`으로 cold-start subset 지표가 산출되지 않았다. 따라서 발표/보고서에서는 다음처럼 정리한다.

```text
Cold-start fallback logic is implemented, but the sampled dev E2E evaluation did not contain cold-start users, so separate cold-start subset metrics were not reported.
```

## 11. Session-aware Recommendation

현재 session-aware 반영 경로:

- candidate training data에 `history_article_ids` 추가
- Two-Tower User Tower가 recent history item embedding average pooling 사용
- serving candidate generation에서 `recent_clicks`와 `session_interest` 반영
- ranking service에서 session signal features 사용
  - `has_recent_click_signal`
  - `has_session_interest_signal`
  - `recent_click_count`
  - `session_interest_count`
  - `session_interest_score`
  - `candidate_reason`

GRU/Transformer 기반 session encoder는 최종 지표 통과에는 사용하지 않았다. 명세를 엄격히 GRU/Transformer 구현으로 해석할 경우 후속 고도화 항목이다.

## 12. 최종 채택안

| 단계 | 최종 채택 |
|---|---|
| Candidate Generation | history-aware Two-Tower + sequential/profile/session/coverage candidate signals |
| Ranking | LogReg CTR ranking pipeline |
| Re-ranking | diversity guard + epsilon-greedy exploration + freshness/new item boost |
| Session | history_article_ids average pooling + recent_click/session_interest serving signals |
| Cold-start | popularity/profile/default-feature fallback |
| Latency | warmup + 75-item serving candidate pool |

최종 artifact:

```text
data/checkpoints/candidate_dev_history_itemid_fast/two_tower.pt
data/checkpoints/candidate_dev_history_itemid_fast/two_tower_metadata.json
rec_models/checkpoints/logreg_dev/ranking_baseline.joblib
rec_models/checkpoints/logreg_dev/ranking_baseline_metadata.json
rec_models/reports/candidate_experiments/dev_two_tower_history_itemid_fast_fixed.json
rec_models/reports/ranking_experiments/dev_ranking_logreg.json
rec_models/reports/baseline/dev_e2e_twotower_serving_latency_pool75_1000.json
rec_models/reports/baseline/dev_serving_latency_top50_50users_pool75.json
```

## 13. 남은 고도화 항목

지표 기준 명세는 통과했지만, production 품질을 더 높이기 위한 후속 과제는 남아 있다.

- GRU/Transformer session encoder 구현
- 온라인 reward update 기반 MAB 고도화
- cold-start 전용 holdout set 구성 및 별도 지표 산출
- full production data 기준 재평가
- Docker image 내부 artifact 경로와 최신 checkpoint 동기화 확인
- 검색 엔진 파트 성능 지표와 통합 리포트 작성

## 14. 문서 갱신 규칙

- 최종 지표를 다시 측정하면 2장, 8장, 9장을 갱신한다.
- Candidate checkpoint가 바뀌면 5장과 최종 artifact 목록을 갱신한다.
- Ranking serving 모델이 바뀌면 6장과 최종 artifact 목록을 갱신한다.
- Cold-start 전용 평가셋 결과가 생기면 10장을 갱신한다.
