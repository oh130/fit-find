# 패션 도메인 Multimodal 검색 & Multi-Stage 추천 시스템

> **FitFind**
> 팀명: 사나이들
> 팀장: 손석범
>
> **팀 구성**
> - 손석범 — 프론트엔드, 평가 대시보드
> - 오승민 — API Gateway, 인프라
> - 이준원 — 데이터 파이프라인
> - 장지원 — 추천 모델 (Two-Tower, Ranking, Re-ranking, MAB)
> - 홍찬근 — 검색 엔진 (CLIP + FAISS)

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 멀티모달 검색 | 텍스트 + 이미지 동시 검색 (CLIP + FAISS HNSW) |
| 개인화 추천 | Two-Tower 후보 생성 → LogReg 랭킹 → MAB 탐색 |
| 페르소나 온보딩 | 9가지 쇼핑 성향 분류 후 Redis 저장, 추천에 반영 |
| 예산 기반 세트 추천 | 예산 내 겹치는 부위 없는 코디 세트 조합 (상의+하의+아우터+신발+액세서리 등) |
| 실시간 세션 반영 | 클릭/장바구니 이벤트 → Redis → 즉시 추천 반영 |
| 평가 대시보드 | 검색 품질 지표, 추천 성능, A/B 테스트 결과 시각화 |

---

## 시스템 아키텍처

```
사용자 (Browser)
       │
       ▼
┌─────────────────────┐
│   Frontend :3000    │  React + Vite + TypeScript
└────────┬────────────┘
         │ HTTP
         ▼
┌─────────────────────────────────────────┐
│         API Gateway :8000               │
│  FastAPI — 단일 진입점                   │
│                                         │
│  POST /api/search          검색          │
│  GET  /api/recommend       개인화 추천   │
│  POST /api/set-recommend   세트 추천     │
│  POST /api/onboarding      페르소나 설정 │
│  POST /api/events          이벤트 기록   │
└────────┬────────────────────┬───────────┘
         │                    │
         ▼                    ▼
┌────────────────┐   ┌────────────────────┐
│ Search Engine  │   │    Rec-Models      │
│    :8002       │   │      :8003         │
│                │   │                    │
│  CLIP 임베딩   │   │  Two-Tower 후보    │
│  FAISS HNSW    │   │  → LogReg 랭킹     │
│  텍스트/이미지  │   │  → Re-ranking      │
│  멀티모달 검색 │   │  → ε-Greedy MAB   │
└────────────────┘   └──────────┬─────────┘
                                │
                     ┌──────────▼─────────┐
                     │    Redis :6379      │
                     │   Feature Store     │
                     │  - recent_clicks    │
                     │  - session_interest │
                     │  - persona profile  │
                     └─────────────────────┘

┌─────────────────────┐   ┌──────────────────────┐
│  Dashboard :8501    │   │  Simulator           │
│  Streamlit          │   │  행동 로그 자동 생성   │
│  - 검색 품질 지표   │   │  - search/view/cart  │
│  - 추천 성능 지표   │   │  - purchase events   │
│  - A/B 테스트 결과  │   └──────────────────────┘
└─────────────────────┘

┌──────────────────────┐
│  CT Pipeline         │
│  성능 모니터링 &      │
│  자동 재학습 트리거   │
└──────────────────────┘
```

---

## 실행 방법

### 사전 요구사항

- Docker Desktop (Docker Compose 포함)
- RAM 16GB 이상 권장
- [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) 데이터셋 (Kaggle)
- Google AI Studio에서 발급한 Gemini API 키

### 1단계: 데이터셋 배치

Kaggle에서 다운로드한 파일을 다음 경로에 배치:

```
data/raw/
├── articles.csv
├── customers.csv
└── transactions_train.csv
```

### 2단계: 환경 변수 설정

`.env.example`을 복사해 `.env`를 생성하고 Gemini API 키를 입력:

```bash
cp .env.example .env
# .env 파일에서 GEMINI_API_KEY= 뒤에 발급받은 키 입력
```

### 3단계: 데이터 파이프라인 실행 (최초 1회)

```bash
docker compose run --rm data-pipeline
```

`dev` 모드로 실행되며 `data/processed/` 아래 학습에 필요한 모든 전처리 파일을 생성합니다. 소요 시간: 약 45~60분.

> 모드 변경이 필요하면 `docker-compose.yml`의 `data-pipeline` 서비스 환경 변수에서 `DATA_PIPELINE_MODE=dev|production`을 수정하세요.

### 4단계: 모델 학습 (최초 1회)

**Two-Tower 후보 모델 학습:**

```bash
docker compose run --rm rec-models python rec_models/candidate/train_two_tower.py
```

**Ranking 모델 학습:**

```bash
docker compose run --rm rec-models python rec_models/ranking/train_ranking.py
```

학습된 체크포인트는 `data/checkpoints/` 아래에 자동 저장됩니다.

### 5단계: 전체 서비스 실행

```bash
docker compose up
```

| 서비스 | 주소 |
|--------|------|
| 프론트엔드 | http://localhost:3000 |
| API Gateway | http://localhost:8000 |
| Search Engine | http://localhost:8002 |
| Rec-Models | http://localhost:8003 |
| 평가 대시보드 | http://localhost:8501 |

---

## API 사용 예시

### 텍스트 검색

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "검정 오버핏 후드티", "top_k": 10}'
```

<details>
<summary>응답 예시</summary>

```json
{
  "search_type": "text",
  "results": [
    {
      "product_id": "0825137001",
      "name": "SABLE denim jacket",
      "score": 0.794,
      "price": 39900
    }
  ],
  "latency_ms": 42.0,
  "total_count": 10
}
```
</details>

### 개인화 추천

```bash
curl "http://localhost:8000/api/recommend?user_id=U1234&top_n=10"
```

<details>
<summary>응답 예시</summary>

```json
{
  "user_id": "U1234",
  "recommendations": [
    {
      "product_id": "0706016001",
      "score": 0.85,
      "reason": "ranking_score",
      "is_exploration": false
    }
  ],
  "pipeline_latency": {
    "candidate_ms": 45,
    "ranking_ms": 62,
    "reranking_ms": 12,
    "total_ms": 127
  }
}
```
</details>

### 예산 기반 세트 추천

```bash
curl -X POST http://localhost:8000/api/set-recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U1234",
    "budget": 150000,
    "num_sets": 3
  }'
```

<details>
<summary>응답 예시</summary>

```json
{
  "sets": [
    {
      "items": [
        {"product_id": "0706016001", "name": "Slim fit shirt", "slot": "top", "price": 29900},
        {"product_id": "0448509014", "name": "Skinny jeans", "slot": "bottom", "price": 49900},
        {"product_id": "0372860001", "name": "Wool coat", "slot": "outer", "price": 59900}
      ],
      "total_price": 139700,
      "within_budget": true
    }
  ]
}
```
</details>

### 이벤트 기록

```bash
curl -X POST http://localhost:8000/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U1234",
    "item_id": "0706016001",
    "event_type": "click",
    "category": "상의"
  }'
```

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 검색 | CLIP (`openai/clip-vit-base-patch32`), FAISS HNSW |
| 추천 | Two-Tower, LogReg Ranking, ε-Greedy MAB |
| LLM 연동 | Google Gemini API (검색 쿼리 이해, 페르소나 분석) |
| 서빙 | FastAPI, Redis, Docker Compose |
| 프론트엔드 | React 18, Vite, TypeScript |
| 대시보드 | Streamlit |
| 데이터 | H&M Personalized Fashion Recommendations (Kaggle, 약 105만 고객 / 3150만 거래) |

---

## 페르소나 시스템

온보딩 시 사용자의 쇼핑 성향을 9가지로 분류하여 Redis에 저장, 추천 파이프라인에 반영합니다.

| 페르소나 | 설명 |
|----------|------|
| `trendsetter` | 트렌드에 민감, 신상 우선 |
| `practical` | 실용성 중심, 기능 중심 구매 |
| `value` | 가성비 중심 |
| `brand_loyal` | 특정 브랜드 선호 |
| `impulse` | 충동 구매 성향 |
| `careful` | 신중한 비교 구매 |
| `repeat_stable` | 재구매 선호, 안정적 패턴 |
| `color_focus` | 색상 중심 선택 |
| `category_focus` | 특정 카테고리 집중 |

