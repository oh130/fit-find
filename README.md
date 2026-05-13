# 멀티모달 검색 및 Multi-Stage 추천 시스템

패션 도메인 지능형 스타일링 서비스 — 2026-1 Capstone Design I  
팀명: 사나이들

---

## 아키텍처

```
사용자 (Browser / API Client)
         │
         ▼
┌─────────────────────┐
│   Frontend :3000    │  React + Vite
└────────┬────────────┘
         │ HTTP
         ▼
┌─────────────────────┐
│  API Gateway :8000  │  FastAPI — 단일 진입점
│  - POST /api/search │
│  - GET  /api/recommend
│  - POST /api/events │
└──┬──────────────┬───┘
   │              │
   ▼              ▼
┌──────────┐  ┌──────────────┐
│  Search  │  │  Rec-Models  │
│  Engine  │  │    :8003     │
│  :8002   │  │              │
│          │  │ Candidate    │
│ CLIP     │  │ Generation   │
│ FAISS    │  │ → Ranking    │
│ HNSW     │  │ → Re-ranking │
└──────────┘  │ → MAB        │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │  Redis :6379 │  Feature Store
              │              │  - recent_clicks
              │              │  - session_interest
              │              │  - click_count
              └──────────────┘

┌─────────────────────┐     ┌──────────────────┐
│  Dashboard :8501    │     │  Simulator       │
│  Streamlit          │     │  행동 로그 생성   │
│  - 검색 품질 지표   │     │  (구현 예정)     │
│  - 추천 성능 지표   │     └──────────────────┘
│  - A/B 테스트 결과  │
└─────────────────────┘
```

---

## 실행 방법

### 사전 요구사항

- Docker Desktop (Docker Compose 포함)
- RAM 16GB 이상 권장

### 전체 시스템 실행

```bash
docker-compose up
```

서비스별 접속 주소:

| 서비스 | 주소 |
|---|---|
| API Gateway | http://localhost:8000 |
| Search Engine | http://localhost:8002 |
| Recommendation | http://localhost:8003 |
| Frontend | http://localhost:3000 |
| Dashboard | http://localhost:8501 |

### 데이터 파이프라인 실행 (최초 1회)

```bash
python data_pipeline/build_article_features.py
python data_pipeline/build_customer_features.py
python data_pipeline/build_item_features.py
python data_pipeline/build_ranking_train_data.py
```

---

## API 사용 예시

### 검색 API

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "검정 오버핏 후드티", "top_k": 10}'
```

응답:

```json
{
  "search_type": "text",
  "results": [
    {"product_id": "0825137001", "name": "SABLE denim jacket", "score": 0.794, "price": 0.0}
  ],
  "latency_ms": 42.0,
  "total_count": 10
}
```

### 추천 API

```bash
curl "http://localhost:8000/api/recommend?user_id=U1234&top_n=10"
```

응답:

```json
{
  "user_id": "U1234",
  "recommendations": [
    {"product_id": "P11111", "score": 0.85, "reason": "ranking_score", "is_exploration": false},
    {"product_id": "P99999", "score": 0.45, "reason": "mab_exploration", "is_exploration": true}
  ],
  "pipeline_latency": {
    "candidate_ms": 45,
    "ranking_ms": 62,
    "reranking_ms": 12,
    "total_ms": 127
  },
  "session_context": {
    "recent_clicks": ["P001", "P002"],
    "session_interest": {"아우터": 3}
  }
}
```

### 이벤트 기록 API

```bash
curl -X POST http://localhost:8000/api/events \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U1234", "item_id": "P001", "event_type": "click", "category": "아우터"}'
```

---

## 팀 구성 및 역할

| 이름 | 담당 파트 |
|---|---|
| 오승민 | API Gateway, Redis Feature Store, Docker Compose, 인프라 |
| 이준원 | 데이터 파이프라인 |
| 홍찬근 | 검색 엔진 (CLIP + FAISS) |
| 장지원 | 추천 모델 (Two-Tower, DeepFM, Re-ranking) |
| 손석범 | 프론트엔드, 평가 대시보드 |

---

## 기술 스택

- **검색**: CLIP (openai/clip-vit-base-patch32), FAISS HNSW
- **추천**: Two-Tower, DeepFM, Epsilon-Greedy MAB
- **서빙**: FastAPI, Redis, Docker Compose
- **프론트엔드**: React, Vite, TypeScript
- **대시보드**: Streamlit
- **데이터**: H&M Personalized Fashion Recommendations (Kaggle)
