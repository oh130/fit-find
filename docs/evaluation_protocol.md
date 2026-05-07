# 추천 모델 평가 프로토콜

## 1. 목적

본 문서는 추천 모델 파트의 실험 및 평가 기준을 일관되게 유지하기 위한 공식 프로토콜을 정의한다. 후보 생성(Candidate Generation), 랭킹(Ranking), 재정렬(Re-ranking), 최종 추천 파이프라인의 성능 비교는 가능한 한 동일한 조건에서 수행한다.

## 2. 적용 범위

본 프로토콜은 `rec_models` 모듈에 적용되며, 다음 단계를 포함한다.

- 후보 생성(Candidate Generation)
- 랭킹(Ranking)
- 재정렬(Re-ranking)
- 최종 추천 파이프라인(Final Recommendation Pipeline)
- 서빙 지연 시간(Serving Latency)

## 3. 사용 데이터

추천 모델 평가는 `data/processed/` 하위의 전처리 데이터를 사용한다.

주요 입력 파일:

- `data/processed/train_data.csv` 또는 실험 기록에 명시된 대체 파일
- `data/processed/customer_features.csv`
- `data/processed/articles_feature.csv`
- `data/processed/item_features.csv`

실험 비교 시에는 동일한 데이터 버전을 사용해야 하며, 입력 파일이 변경된 경우 반드시 실험 기록에 명시한다.

## 4. 데이터 분할 기준

추천 모델 평가는 고정된 데이터 분할 기준에서 수행한다.

- 학습 데이터: `train`
- 검증 데이터: `validation`
- 최종 평가 데이터: `test`

가능한 경우 시간 순 분할을 우선 적용한다. 랜덤 분할을 사용할 경우에는 다음 항목을 반드시 기록한다.

- 분할 방식
- 분할 비율
- random seed

공식 결과 비교는 동일한 split 기준에서만 수행한다.

## 5. Random Seed

재현 가능한 실험을 위해 공식 random seed는 다음과 같이 고정한다.

- `seed = 42`

가능한 경우 다음 항목에 동일한 seed를 적용한다.

- 데이터 분할
- negative sampling
- candidate sampling
- 학습 초기화
- reranking 관련 무작위 선택

## 6. 평가 유저 범위

평가 유저 범위는 아래와 같이 구분한다.

- 개발용 점검: `max_users=100`
- 중간 점검: `max_users=1000`
- 공식 결과: 전체 평가 유저, `max_users=None`

공식 비교 결과와 보고서용 표는 전체 평가 유저 기준 결과를 우선 사용한다.

## 7. 후보 생성 평가 기준

후보 생성 단계는 다음 지표로 평가한다.

- `Recall@300`

주요 비교 대상:

- popularity / metadata 기반 baseline candidate generation
- Two-Tower candidate generation

후보 생성 실험은 동일한 유저 집합, 동일한 평가 데이터, 동일한 top-k 조건에서 비교한다.

## 8. 랭킹 평가 기준

랭킹 단계는 다음 지표로 평가한다.

- `AUC`

주요 비교 대상:

- Logistic Regression baseline
- DeepFM ranking model

랭킹 비교는 가능한 한 동일한 candidate pool 조건에서 수행한다.

## 9. 최종 추천 평가 기준

최종 추천 파이프라인은 다음 지표로 평가한다.

- `HitRate@50`
- `NDCG@50`
- `Coverage@50`
- serving latency

주요 비교 대상:

- Popularity baseline
- Current serving baseline
- 개선된 추천 파이프라인

Latency는 warmup 이후 실제 serving orchestration 경로의 요청 단위 지연 시간으로 측정한다. 서버 시작 시 model/checkpoint/artifact를 메모리에 올리는 warmup 비용은 request latency에서 제외한다.

Latency 목표:

- 평균 latency <= 200ms
- p95 latency <= 200ms를 권장 기준으로 함께 기록한다.

측정 도구:

- `rec_models.evaluation.measure_serving_latency`

## 10. 재정렬(Re-ranking) 평가 기준

재정렬 단계에서는 다음 설정들의 효과를 비교한다.

- reranking 없음
- diversity만 적용
- diversity + exploration 적용
- diversity + exploration + freshness 적용

비교 지표:

- `HitRate@50`
- `NDCG@50`
- `Coverage@50`
- latency 변화

재정렬 실험의 목적은 정확도와 다양성 간 trade-off를 확인하는 데 있다.

## 11. Cold-start 평가

Cold-start 사용자 성능은 전체 결과와 별도로 측정한다.

Cold-start 사용자는 다음 중 하나를 만족하는 경우로 정의한다.

- 사용자 feature가 없음
- 행동 이력이 부족함

평가 지표:

- `HitRate@50`
- `NDCG@50`
- `Coverage@50`

## 12. Baseline 정의

추천 모델 비교를 위한 공식 baseline은 다음 두 가지로 정의한다.

### Baseline A

- popularity-only recommendation

### Baseline B

- 현재 서빙 baseline 파이프라인
- metadata 및 session 기반 candidate generation
- Logistic Regression ranking
- reranking 적용

## 13. 결과 저장 규칙

실험 결과는 원본 결과와 사람이 읽는 요약 문서를 분리하여 저장한다.

원본 결과 저장 위치:

- `rec_models/reports/baseline/`
- `rec_models/reports/candidate_experiments/`
- `rec_models/reports/ranking_experiments/`
- `rec_models/reports/reranking_experiments/`
- `rec_models/reports/final/`

요약 문서 위치:

- `docs/recommendation_experiments.md`

실험 수행 절차와 명령어 정리는 다음 문서를 따른다.

- `docs/recommendation_experiment_workflow.md`

권장 파일 형식:

- `.json`
- `.md`

## 14. 파일명 규칙

실험 결과 파일명은 다음 규칙을 따른다.

- `YYYY-MM-DD_<experiment_name>.json`
- `YYYY-MM-DD_<experiment_name>.md`

예시:

- `2026-05-03_baseline_full.json`
- `2026-05-03_two_tower_v2.json`
- `2026-05-03_deepfm_v1.json`

## 15. 공식 결과 인정 기준

공식 비교 결과로 인정하려면 다음 조건을 만족해야 한다.

- 고정된 데이터 버전 사용
- 고정된 split 기준 사용
- 고정된 random seed 사용
- 공식 metric set 포함
- latency 목표가 있는 실험은 warmup 여부와 측정 기준 포함
- 실험 날짜 및 설정 기록
- 적절한 report 디렉토리에 결과 저장 완료

## 16. 비고

개발 중 빠른 확인을 위한 샘플 실험은 허용하되, 공식 최종 비교 결과로는 사용하지 않는다. 모델 구조나 feature schema가 크게 바뀐 경우에는 동일 프로토콜로 baseline을 다시 측정한다.

`docs/recommendation_experiments.md`에는 모든 raw 실험 결과를 다 기록하지 않는다. 해당 문서에는 다음에 해당하는 핵심 실험만 요약하여 누적 기록한다.

- baseline 비교 기준이 되는 실험
- 모델 선택에 영향을 준 실험
- 최종 채택안과 직접 관련 있는 실험
