# 추천 모델 실험 진행 방법

## 1. 목적

본 문서는 추천 모델 파트에서 baseline, 후보 생성, 랭킹, 재정렬 실험을 어떤 순서로 수행하고 결과를 어디에 저장할지 정리한 작업용 문서이다.  
공식 평가 기준은 `docs/evaluation_protocol.md`, 핵심 결과 요약은 `docs/recommendation_experiments.md`를 따른다.

## 2. 문서와 결과 파일의 역할

- `docs/evaluation_protocol.md`
  - 평가 기준, 공식 metric, baseline 정의
- `docs/recommendation_experiments.md`
  - 핵심 실험 결과만 요약
- `rec_models/reports/...`
  - 각 실험의 raw 결과 저장

즉, 실험을 실행할 때마다 먼저 `rec_models/reports/`에 JSON 또는 Markdown 결과를 남기고, 그중 의미 있는 결과만 `docs/recommendation_experiments.md`에 반영한다.

## 3. 실험 진행 순서

추천 모델 실험은 아래 순서로 진행한다.

1. baseline 전체 성능 확인
2. Candidate Generation 실험
3. Ranking 실험
4. Re-ranking 튜닝 실험
5. Cold-start 성능 확인
6. 최종 채택안 정리

## 4. Baseline 실험

현재 추천 서빙 baseline 전체 성능을 확인하는 단계이다.

실행 예시:

```bash
python -m rec_models.evaluation.evaluate_recommender \
  --data data/processed/train_data_test.csv \
  --top_k 50 \
  --max-users 100 \
  --output-json rec_models/reports/baseline/2026-05-03_baseline_dev.json
```

공식 결과 저장 예시:

```bash
python -m rec_models.evaluation.evaluate_recommender \
  --data data/processed/train_data.csv \
  --top_k 50 \
  --output-json rec_models/reports/baseline/2026-05-03_baseline_full.json
```

확인할 지표:

- `HitRate@50`
- `NDCG@50`
- `Coverage@50`
- popularity baseline 대비 차이

## 5. Candidate Generation 실험

현재 candidate 실험은 아래 순서로 진행한다.

1. sequential transition artifact 생성
2. history-aware diagnostic으로 `baseline/profile/copurchase/sequential` 비교
3. sequential_combined 기준으로 `include_two_tower=False` vs `True` 비교
4. 최종적으로 serving default 조합 갱신

현재 기본 serving candidate는 `sequential_combined + Two-Tower hybrid`이다.

artifact 생성 예시:

```bash
python -m rec_models.candidate.build_sequential_transitions \
  --exclude-last-per-user
```

history-aware 비교 예시:

```bash
python -m rec_models.evaluation.diagnose_recommender candidate \
  --history-aware \
  --history-aware-two-tower-comparison \
  --data data/processed/train_data_test.csv \
  --max-users 100 \
  --sample-size 100 \
  --output-dir /tmp/rec_diag_candidate_sequential_100
```

확인할 지표:

- `Recall@300`

중점 확인 사항:

- sequential_category / sequential_combined가 baseline 대비 개선되는지
- `include_two_tower=False` 대비 `include_two_tower=True` hybrid가 추가 이득이 있는지
- leakage-safe artifact 생성 여부가 metadata에 남는지

## 6. Ranking 실험

현재는 Logistic Regression baseline 평가가 가능하며, 이후 DeepFM 실험도 같은 항목에 누적한다.

실행 예시:

```bash
python -m rec_models.ranking.evaluator \
  --data data/processed/train_data_test.csv \
  --top_k 50 \
  --max-users 100 \
  --output-json rec_models/reports/ranking_experiments/2026-05-03_ranking_baseline_dev.json
```

확인할 지표:

- `AUC`
- `HitRate@50`
- `NDCG@50`

## 7. Re-ranking 실험

재정렬 로직의 효과를 비교하는 단계이다. 현재는 별도 자동화 스크립트가 충분히 정리되지 않았으므로, 실험 조건을 명확히 적고 결과를 수동 정리한다.

비교할 설정:

- reranking 없음
- diversity only
- diversity + exploration
- diversity + exploration + freshness

현재 단계에서는 결과 파일을 `rec_models/reports/reranking_experiments/`에 저장하고, 핵심 비교 결과만 `docs/recommendation_experiments.md`에 반영한다.

## 8. Cold-start 성능 확인

Cold-start 사용자는 전체 성능과 별도로 확인한다.

확인 항목:

- `HitRate@50`
- `NDCG@50`
- `Coverage@50`

현재 `evaluate_recommender.py` 결과에는 `cold_start_subset` 항목이 포함되므로, 별도 지표를 함께 확인한다.

## 9. 문서 반영 규칙

아래 경우에만 `docs/recommendation_experiments.md`를 갱신한다.

- baseline 기준 결과가 갱신된 경우
- Two-Tower 최고 성능이 갱신된 경우
- DeepFM 실험 결과가 처음 확보된 경우
- reranking 비교 결과가 정리된 경우
- 최종 채택 모델이 바뀐 경우

즉, 모든 실험 결과를 다 옮기지 말고, 의미 있는 결과만 남긴다.

## 10. 권장 작업 흐름

실무적으로는 아래 순서로 움직이면 된다.

1. 실험 실행
2. `rec_models/reports/...`에 결과 저장
3. 이전 결과와 비교
4. 의미 있는 변화가 있으면 `docs/recommendation_experiments.md` 갱신
5. 최종적으로 채택된 결과만 별도 정리
