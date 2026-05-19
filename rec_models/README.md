# 추천 서비스 (`rec_models`)

`rec_models`는 본 프로젝트의 추천 시스템 서비스 모듈입니다. FastAPI 기반 추천 서버를 통해 상품 추천 결과를 제공하며, 후보 생성부터 랭킹, 리랭킹, 오프라인 평가, Docker 단독 실행까지 포함합니다.

현재 기준으로 구현이 완료된 범위는 다음과 같습니다.

- candidate generation
- ranking inference
- reranking
- offline evaluation
- API serving
- Docker standalone execution

## 1. 프로젝트 개요

### `rec_models`의 역할

`rec_models` 서비스는 사용자 요청에 대해 최종 추천 결과를 생성하는 역할을 담당합니다.

주요 책임은 다음과 같습니다.

- 추천 후보군 생성
- 랭킹 모델 기반 후보 점수화
- 다양성, 신선도, 탐색을 반영한 리랭킹
- 추천 이유와 지연 시간 정보를 포함한 API 응답 반환
- 현재 추천 파이프라인에 대한 오프라인 평가 지원

### 전체 추천 파이프라인

현재 추천 서빙 흐름은 다음과 같습니다.

1. Candidate Generation
   - 기본 경로에서는 sequential transition, 상품 메타데이터, popularity, recent clicks, session interest, Two-Tower retrieval을 결합해 후보군을 생성합니다.
   - 사용자/세션 신호가 부족한 경우 popularity 기반 fallback을 사용합니다.
2. Ranking
   - `joblib`로 저장된 baseline ranking 모델로 후보군을 점수화합니다.
   - 사용자 프로필 feature와 상품 메타데이터를 결합해 ranking feature를 구성합니다.
3. Reranking
   - 카테고리 다양성을 반영해 결과를 조정합니다.
   - exploration slot을 삽입합니다.
   - 필요 시 신상품/신선도 관련 boost를 적용합니다.
4. Response
   - 최종 추천 결과와 함께 `product_id`, `score`, `reason`, `is_exploration`, latency 정보를 반환합니다.

요약하면 다음과 같습니다.

```text
Request
  -> Candidate Generation
  -> Ranking
  -> Reranking
  -> Final Recommendations API Response
```

## 2. 주요 기능

### Candidate Generation

- `serving/candidate_service.py`에 메타데이터 기반 candidate generation이 구현되어 있습니다.
- popularity 기반 fallback candidate retrieval을 지원합니다.
- 기본 serving candidate 전략은 `sequential_combined + Two-Tower hybrid`입니다.
- sequential transition artifact를 사용합니다.
  - `article_id -> next_article_id`
  - `main_category -> next_article_id`
- co-purchase artifact는 남아 있지만 기본 serving 경로에서는 사용하지 않습니다.
- 세션 기반 후보 확장을 지원합니다.
  - `recent_clicks`
  - `session_interest`
- 최근 클릭한 아이템은 최종 candidate pool에서 제외합니다.
- `candidate_reason`, popularity, freshness 관련 메타데이터를 함께 생성합니다.

기본 설정은 다음과 같습니다.

- `include_sequential=True`
- `include_two_tower=True`
- `include_copurchase=False`

현재 candidate 단계에서 동작하는 주요 분기는 다음과 같습니다.

- `cold_start_popularity`
  - recent clicks와 session interest가 모두 없을 때 사용됩니다.
- `sequential_article_transition`
  - 최근 구매/클릭 article에서 다음 1~3 step 전이 후보를 확장합니다.
- `sequential_category_transition`
  - 최근 구매/클릭 article의 `main_category`에서 다음 1~3 step 전이 후보를 확장합니다.
- `two_tower_candidate`
  - Two-Tower retrieval 후보를 hybrid 방식으로 상단 candidate pool에 우선 반영합니다.
- recent-click signal matching
  - 최근 클릭 상품과의 category / main category / color 유사도를 활용합니다.
- session-interest matching
  - 세션 관심도 가중치를 category 단위로 반영합니다.

### Ranking

- `serving/ranking_service.py`에 baseline ranking inference가 구현되어 있습니다.
- `checkpoints/ranking_baseline.joblib`에 저장된 sklearn pipeline을 로드합니다.
- `joblib` artifact와 metadata를 함께 사용합니다.
- 다음 feature를 결합해 ranking input을 구성합니다.
  - customer profile features
  - item metadata
  - engineered cross features
- 사용자 feature가 없는 경우 cold-start-safe default 값을 사용합니다.

### Reranking

- `serving/rerank_bridge.py`에 reranking 로직이 구현되어 있습니다.
- 가능한 경우 동일 카테고리 아이템이 3개 연속 등장하지 않도록 diversity guard를 적용합니다.
- ranking 결과 중 일부 위치에 exploration slot을 삽입합니다.
- exploration 과정에서 fresh/new item boost를 적용할 수 있습니다.
- ranking 단계가 실패하면 popularity 기반 정렬로 fallback합니다.

현재 적용된 reranking 기능은 다음과 같습니다.

- diversity control
- exploration slot injection
- freshness fallback / new item boost
- reward-aware epsilon-greedy exploration policy
- Redis 기반 item-level reward update와 UCB exploration score

### `reason` 필드 분기

최종 추천 결과에는 `reason` 필드가 포함됩니다. 현재 사용 중인 분기는 다음과 같습니다.

- `recent_click_similarity`
- `session_interest_match`
- `cold_start_popularity`
- `ranking_score`
- `new_item_boost`
- `mab_exploration`
- `bandit_reward_exploration`

### Latency 측정

각 요청에 대해 파이프라인 단계별 latency를 측정해 반환합니다.

- `candidate_ms`
- `ranking_ms`
- `reranking_ms`
- `total_ms`

### API 서버

구현된 엔드포인트는 다음과 같습니다.

- `GET /recommend`
- `POST /session/update`
- `GET /health`
- Swagger 문서: `/docs`

### Evaluation CLI

오프라인 평가는 `evaluation/evaluate_recommender.py`에 구현되어 있습니다.

지원하는 지표는 다음과 같습니다.

- `HitRate@K`
- `NDCG@K`
- `Coverage@K`

추가로 다음 기능도 지원합니다.

- cold-start subset 평가
- popularity baseline 비교
- JSON 결과 저장

### Docker 단독 실행

- `rec_models`는 단독 Docker 서비스로 실행할 수 있습니다.
- Docker 이미지에는 FastAPI 서버와 `rec_models/checkpoints/logreg_dev` ranking artifact를 포함할 수 있습니다.
- Two-Tower candidate artifact는 repo-root `data/checkpoints` 아래에 두고 `/app/data`로 마운트하는 실행 방식을 기준으로 합니다.

## 3. 디렉토리 구조

`rec_models`의 주요 디렉토리는 다음과 같습니다.

```text
rec_models/
├── serving/
├── ranking/
├── evaluation/
├── checkpoints/
└── data/processed/
```

### `serving/`

추천 서빙 시점의 핵심 로직이 들어 있습니다.

- `candidate_service.py`
  - candidate generation 로직
- `ranking_service.py`
  - ranking inference 및 feature 구성
- `rerank_bridge.py`
  - diversity, exploration, freshness 기반 reranking
- `recommend_service.py`
  - candidate -> ranking -> reranking 전체 orchestration

### `ranking/`

랭킹 모델 학습 및 추론 관련 유틸리티가 포함됩니다.

- dataset preparation
- model training
- offline ranking inference helpers
- ranking model evaluator utilities

### `evaluation/`

추천 서비스 오프라인 평가 도구가 포함됩니다.

- serving pipeline evaluation CLI
- recommendation metrics
- popularity baseline comparison

### `checkpoints/`

학습된 모델 artifact와 metadata를 저장합니다.

- `logreg_dev/ranking_baseline.joblib`
  - 최종 서빙에 사용하는 LogReg CTR ranking pipeline
- `logreg_dev/ranking_baseline_metadata.json`
  - 최종 ranking pipeline과 함께 사용하는 feature metadata
- `ranking_baseline.joblib`, `ranking_baseline_metadata.json`
  - 이전 baseline fallback artifact
- `two_tower.pt`
  - Two-Tower candidate 모델 checkpoint
- `deepfm.pt`
  - DeepFM ranking 실험 checkpoint. 최종 serving 채택 모델은 LogReg CTR ranker이다.

### `data/processed/`

서빙 및 평가에 사용하는 전처리 결과 파일이 들어 있습니다.

- `articles_feature.csv`
  - 상품/아이템 메타데이터
- `customer_features.csv`
  - 사용자 프로필 feature
- `item_features.csv`
  - popularity 등 item-level feature

## 4. 실행 방법

### 4.1 로컬 실행

레포지토리 루트에서 터미널을 열고 실행합니다.

- 권장 환경
  - WSL terminal
  - VSCode integrated terminal

프로젝트 루트 기준 실행:

```bash
cd /home/jiwon/projects/multimodal-search-engine
python -m venv .venv
source .venv/bin/activate
pip install -r rec_models/requirements.txt
python -m uvicorn rec_models.app:app --host 0.0.0.0 --port 8003
```

로컬 실행 전 최종 서빙 artifact가 있어야 합니다.

```text
rec_models/checkpoints/logreg_dev/ranking_baseline.joblib
rec_models/checkpoints/logreg_dev/ranking_baseline_metadata.json
data/checkpoints/candidate_dev_history_itemid_fast/two_tower.pt
data/checkpoints/candidate_dev_history_itemid_fast/two_tower_metadata.json
```

Ranking checkpoint는 `RANKING_CHECKPOINT_DIR` 환경변수 경로에 artifact가 있으면 그 경로를 사용합니다. 없으면 `rec_models/checkpoints/logreg_dev`, `rec_models/checkpoints` 순서로 fallback합니다. Candidate checkpoint는 `TWO_TOWER_CHECKPOINT_DIR` 환경변수 경로에 `two_tower.pt`가 있으면 그 경로를 사용합니다. 없으면 `data/checkpoints/candidate_dev_history_itemid_fast`, `data/checkpoints/candidate_dev_history_lolo_fast`, `data/checkpoints/candidate` 순서로 fallback합니다.

artifact가 없다면 팀 공유 산출물을 위 경로에 배치하거나, dev 데이터 기준으로 아래 순서대로 생성합니다.

```bash
DATA_PIPELINE_MODE=dev .venv/bin/python data_pipeline/run_data_pipeline.py

.venv/bin/python -m rec_models.candidate.train \
  --data data/processed/candidate_train_data_dev.csv.gz \
  --epochs 2 \
  --batch-size 1024 \
  --validation-max-users 1000 \
  --checkpoint-dir data/checkpoints/candidate_dev_history_itemid_fast

.venv/bin/python -m rec_models.ranking.train \
  --model-type logreg \
  --split-mode user \
  --output-dir rec_models/checkpoints/logreg_dev
```

서버 주소:

```text
http://localhost:8003
```

API 문서:

```text
http://localhost:8003/docs
```

### 4.2 Docker 실행

레포지토리 루트에서 아래 명령으로 빌드 및 실행합니다.

```bash
cd /home/jiwon/projects/multimodal-search-engine
docker build -t rec-models ./rec_models
docker run --rm -p 8003:8003 -v "$PWD/data:/app/data:ro" rec-models
```

Docker build context는 `./rec_models`이므로 `rec_models/checkpoints/logreg_dev`에 최신 ranking artifact가 있으면 이미지에 포함됩니다. 해당 dev artifact가 없으면 `rec_models/checkpoints`의 root ranking artifact로 fallback합니다. Candidate Two-Tower artifact는 repo-root `data/checkpoints/candidate_dev_history_itemid_fast`에 둔 뒤 위처럼 `/app/data`로 마운트합니다. 해당 경로가 없으면 `candidate_dev_history_lolo_fast`, `candidate` 순서로 fallback합니다. 전체 `docker-compose.yml` 실행도 같은 `/app/data` 마운트를 사용합니다.

실행 후 접속 주소:

```text
http://localhost:8003
http://localhost:8003/docs
```

## 5. API 사용 방법

### `GET /recommend`

사용자에 대한 추천 결과를 반환합니다.

Query parameter:

- `user_id` (required)
- `top_n` (optional, default: `10`)
- `recent_clicks` (optional, comma-separated article ids)
- `click_count` (optional, default: `0`)
- `session_interest` (optional, JSON string)

예시 요청:

```bash
curl "http://localhost:8003/recommend?user_id=12345&top_n=10&recent_clicks=0108775015,0751471001&click_count=2&session_interest=%7B%22Dresses%22%3A0.8%2C%22Tops%22%3A0.4%7D"
```

예시 응답 형태:

```json
{
  "user_id": "12345",
  "recommendations": [
    {
      "product_id": "0751471001",
      "score": 0.9132,
      "reason": "recent_click_similarity",
      "is_exploration": false
    },
    {
      "product_id": "0861234002",
      "score": 0.8411,
      "reason": "new_item_boost",
      "is_exploration": true
    }
  ],
  "pipeline_latency": {
    "candidate_ms": 85,
    "ranking_ms": 41,
    "reranking_ms": 3,
    "total_ms": 129
  },
  "session_context": {
    "recent_clicks": ["0108775015", "0751471001"],
    "session_interest": {
      "Dresses": 0.8,
      "Tops": 0.4
    }
  }
}
```

### `POST /session/update`

추천 노출 이후 발생한 클릭/장바구니/구매 이벤트를 reward-aware exploration에 반영합니다. Redis가 사용 가능하면 item-level reward 통계를 갱신하고, Redis가 없으면 추천 API 안정성을 위해 no-op으로 성공 응답을 반환합니다.

예시 요청:

```bash
curl -X POST "http://localhost:8003/session/update" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "12345",
    "item_id": "0751471001",
    "event": "click"
  }'
```

예시 응답:

```json
{
  "status": "ok",
  "bandit_updated": true,
  "article_id": "0751471001",
  "event": "click",
  "reward": 1.0
}
```

### `GET /health`

서버 상태 확인용 health check 엔드포인트입니다.

예시 요청:

```bash
curl "http://localhost:8003/health"
```

예시 응답:

```json
{
  "status": "ok"
}
```

### Interactive API Docs

서버가 실행 중이면 아래 주소에서 Swagger UI를 확인할 수 있습니다.

```text
http://localhost:8003/docs
```

## 6. Evaluation 방법

### CLI 실행

레포지토리 루트에서 아래 명령으로 오프라인 평가를 실행합니다.

```bash
.venv/bin/python -m rec_models.evaluation.evaluate_recommender \
  --top_k 50 \
  --candidate-k 300 \
  --max-users 1000 \
  --use-serving-candidates \
  --output-json rec_models/reports/baseline/dev_e2e_twotower_serving_latency_pool75_1000.json
```

추가 예시는 다음과 같습니다.

```bash
.venv/bin/python -m rec_models.evaluation.evaluate_recommender \
  --top_k 50 \
  --candidate-k 300 \
  --max-users 1000 \
  --use-serving-candidates \
  --skip-popularity-baseline \
  --output-json rec_models/reports/baseline/dev_e2e_twotower_serving_latency_pool75_1000.json
```

```bash
.venv/bin/python -m rec_models.evaluation.measure_serving_latency \
  --top-k 50 \
  --max-users 50 \
  --output-json rec_models/reports/baseline/dev_serving_latency_top50_50users_pool75.json
```

### 지표 설명

- `HitRate@K`
  - 상위 `K`개 추천 안에 relevant item이 하나라도 포함되는지 측정합니다.
- `NDCG@K`
  - relevant item이 상위에 올수록 더 높은 점수를 주는 ranking 품질 지표입니다.
- `Coverage@K`
  - 추천 결과가 전체 candidate item 공간을 얼마나 넓게 사용하는지 측정합니다.

### Popularity Baseline 비교

Evaluation CLI는 현재 serving pipeline과 popularity-only baseline을 같은 candidate set 위에서 비교할 수 있습니다.

이를 통해 다음을 확인할 수 있습니다.

- ranking + reranking이 relevance를 실제로 개선하는지
- diversity / exploration이 coverage에 어떤 영향을 주는지
- cold-start 상황에서 popularity-only 정렬보다 나은지

CLI는 다음 결과를 함께 출력합니다.

- current model metrics
- cold-start subset metrics
- popularity baseline metrics
- improvement versus popularity baseline

### 최종 dev 기준 결과

최신 dev mode 기준 추천 모델은 명세 지표를 통과했습니다. 아래 값은 2026-05-13 생성 dev 데이터와 9개 페르소나 score 기준으로 2026-05-14 재학습/재평가한 결과입니다.

| 항목 | 기준 | 결과 | 판정 |
|---|---:|---:|---|
| Candidate Recall@300 | >= 0.30 | 0.602922 | 통과 |
| Ranking AUC | >= 0.70 | 0.960505 | 통과 |
| HitRate@50 | >= 0.20 | 0.491000 | 통과 |
| NDCG@50 | >= 0.08 | 0.177184 | 통과 |
| Coverage@50 | >= 0.20 | 0.214607 | 통과 |
| Serving latency p95 | <= 200ms | 162.23ms | 통과 |

동일 1000 유저 기준 popularity baseline 대비 개선폭:

| 항목 | Current Model | Popularity Baseline | Improvement |
|---|---:|---:|---:|
| HitRate@50 | 0.491000 | 0.405000 | +0.086000 |
| NDCG@50 | 0.177184 | 0.068926 | +0.108258 |
| Coverage@50 | 0.214228 | 0.015075 | +0.199153 |

5000명 확대 평가에서도 HitRate@50 `0.480800`, NDCG@50 `0.173457`, Coverage@50 `0.536535`로 명세를 통과했습니다.

주요 결과 파일:

- `rec_models/reports/candidate_experiments/dev_two_tower_persona_latest_1000.json`
- `rec_models/reports/ranking_experiments/dev_ranking_logreg_persona_latest_1000.json`
- `rec_models/reports/baseline/dev_e2e_persona_latest_1000.json`
- `rec_models/reports/baseline/dev_e2e_persona_latest_baseline_compare_1000.json`
- `rec_models/reports/baseline/dev_e2e_persona_latest_5000.json`
- `rec_models/reports/baseline/dev_serving_latency_persona_latest_top50_50users.json`

## 7. 현재 한계

- dev 기준 명세 지표는 통과했지만, full production data 기준 재검증은 별도 수행이 필요합니다.
- cold-start fallback은 구현되어 있으나 dev E2E 샘플에서 cold-start subset이 0명이라 별도 subset metric은 없습니다.
- 데이터셋 구조와 positive label 추론 방식 때문에 평가 편향이 발생할 수 있습니다.
- reward-aware exploration은 item-level UCB 기반이며, user-context별 posterior를 학습하는 contextual bandit은 아닙니다.
- Session은 recent history/session signal 기반이며, GRU/Transformer session encoder는 후속 고도화 항목입니다.

## 8. TODO

- full production data 기준 최종 재평가
- Docker image 내부 artifact가 최신 Two-Tower/logreg checkpoint와 동기화되는지 배포 전 확인
- cold-start 전용 holdout set 구성 및 별도 metric 산출
- GRU/Transformer 기반 session encoder 고도화
- popularity fallback을 넘어서는 cold-start 전략 고도화
- reward-aware exploration의 offline/online lift 검증
- user embedding / item embedding 등 feature 추가
- train/test split 전략 개선으로 offline evaluation 신뢰도 향상
- volume mount 또는 external storage 기반 대용량 데이터 처리 개선
- `docker-compose` 기반 전체 서비스 통합

## Current Status Summary

- FastAPI 기반 추천 API 서버가 구현되어 있습니다.
- candidate generation, ranking, reranking이 end-to-end로 연결되어 있습니다.
- dev 기준 Candidate Recall, Ranking AUC, HitRate, NDCG, Coverage, latency 명세를 통과했습니다.
- 오프라인 evaluation CLI에서 HitRate, NDCG, Coverage 및 popularity baseline 비교를 지원합니다.
- Docker standalone 실행이 가능하며 포트는 `8003`을 사용합니다.
- 다음 우선 과제는 production data 재검증, cold-start subset 평가, session/MAB 고도화입니다.
