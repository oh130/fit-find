# Serving

서비스 환경에서 추천 추론 흐름을 연결하는 계층이다.

## 역할
- 저장된 candidate 모델, ranking 모델, session feature를 묶어 최종 추천을 반환한다.
- 학습 코드와 분리된 추론 전용 서비스 인터페이스를 제공한다.

## 구현 범위
- candidate model 로드 및 retrieval 호출 정의
- ranking model 로드 및 scoring 호출 정의
- 전체 recommend 파이프라인 진입점 정의
- rerank 또는 정책 모듈과 연결되는 브릿지 정의

현재 기본 candidate 조합은 다음과 같다.

- sequential article transition
- sequential category transition
- Two-Tower retrieval hybrid
- user profile / recent click / session interest signals
- coverage exploration candidates
- co-purchase는 실험용 opt-in

## 현재 채택 설정

2026-05-06 dev 평가 기준으로 serving 추천 파이프라인은 다음 설정을 사용한다.

- Candidate checkpoint:
  - `data/checkpoints/candidate_dev_history_itemid_fast/two_tower.pt`
- Ranking checkpoint:
  - `rec_models/checkpoints/logreg_dev/ranking_baseline.joblib`
- Serving candidate pool:
  - top-50 요청 기준 75개 후보를 생성해 ranking/reranking에 전달
- Reranking:
  - category diversity guard
  - epsilon-greedy exploration slot
  - freshness/new item boost
  - coverage exploration priority
  - optional priority weights: `persona_hint`, `personalization_weight`, `popularity_weight`, `price_weight`, `diversity_weight`, `freshness_weight`, `exploration_weight`, `long_tail_weight`

우선순위 가중치는 `0.0`~`5.0` 범위로 받는다. `personalization_weight`와 `popularity_weight`는 최종 reranking에서 모델 개인화 점수와 인기 점수의 혼합 비율을 조절한다. `persona_hint`는 카테고리 관심도 floor와 페르소나별 가격/다양성/신상품/탐색 보정 profile을 함께 적용한다. API Gateway는 `/api/recommend` 쿼리 파라미터를 rec-models `/recommend`로 그대로 전달한다.

검증 결과:

```text
Candidate Recall@300 = 0.614632
Ranking AUC          = 0.956739
HitRate@50           = 0.497000
NDCG@50              = 0.178138
Coverage@50          = 0.215203
Latency p95          = 187.61ms
```

주요 report:

- `rec_models/reports/candidate_experiments/dev_two_tower_history_itemid_fast_fixed.json`
- `rec_models/reports/ranking_experiments/dev_ranking_logreg.json`
- `rec_models/reports/baseline/dev_e2e_twotower_serving_latency_pool75_1000.json`
- `rec_models/reports/baseline/dev_serving_latency_top50_50users_pool75.json`

## 요구사항
- 학습 스크립트를 직접 호출하지 않아야 한다.
- 단계별 실패나 빈 결과에 대한 fallback 여지를 고려해야 한다.
- 응답 지연을 고려한 단순한 추론 인터페이스를 유지해야 한다.

## 제외 범위
- 실제 모델 학습
- 데이터셋 생성과 오프라인 실험 관리
- 외부 게이트웨이 수준의 통합 API 조정
