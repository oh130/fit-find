# API Gateway

멀티모달 검색 및 추천 시스템의 진입점 서버.  
모든 클라이언트 요청을 받아 내부 서비스로 프록시하고, Redis 기반 사용자 세션을 관리한다.

---

## 역할

- 검색 요청 → search-engine(8002) 프록시
- 추천 요청 → Redis 세션 데이터를 붙여 rec-models(8003) 프록시
- 클릭/구매 이벤트 → Redis에 저장 및 rec-models 세션 업데이트
- 사용자 피처(최근 클릭, 세션 관심사, 클릭 수) → Redis 조회

---

## 포트

| 서비스 | 포트 |
|--------|------|
| API Gateway | 8000 |
| search-engine | 8002 |
| rec-models | 8003 |
| Redis | 6379 |

---

## 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/api/search` | 검색 요청 프록시 |
| GET | `/api/recommend` | 추천 요청 프록시 |
| POST | `/api/events` | 클릭/구매 이벤트 저장 |
| GET | `/api/features/{user_id}` | 사용자 피처 조회 |
| GET | `/health` | 서버 및 Redis 상태 확인 |

### POST /api/search

```json
// 요청
{
  "query": "흰색 반팔",
  "image_base64": null,
  "top_k": 10
}

// 응답 (search-engine 반환값 그대로)
{
  "search_type": "text",
  "results": [{ "product_id": "...", "name": "...", "score": 0.95, "price": 29000.0 }],
  "latency_ms": 12.3,
  "total_count": 10
}
```

### GET /api/recommend

```
GET /api/recommend?user_id=U1234&top_n=10
```

추천 우선순위 조절 파라미터를 함께 전달할 수 있다. 범위는 `0.0`~`5.0`이며, 미전달 또는 `1.0`은 기본 추천 정책과 동일하다.

- `price_weight`: 높을수록 낮은 가격 상품을 더 선호
- `popularity_weight`: 높을수록 인기 상품 쏠림을 더 강하게 완화
- `diversity_weight`: 높을수록 카테고리 다양성 강화
- `freshness_weight`: 높을수록 신상품 boost 강화
- `exploration_weight`: 높을수록 MAB/coverage exploration 슬롯 강화

예시:

```
GET /api/recommend?user_id=U1234&top_n=10&price_weight=1.5&diversity_weight=2.0
```

```json
// 응답
{
  "user_id": "U1234",
  "recommendations": [
    { "product_id": "...", "score": 0.87, "reason": "ranking_score", "is_exploration": false }
  ],
  "pipeline_latency": {
    "candidate_ms": 30,
    "ranking_ms": 20,
    "reranking_ms": 10,
    "total_ms": 60
  },
  "session_context": {
    "recent_clicks": ["item_001", "item_002"],
    "session_interest": { "상의": 3, "하의": 1 }
  }
}
```

### POST /api/events

```json
// 요청
{
  "user_id": "U1234",
  "item_id": "item_001",
  "event_type": "click",
  "category": "상의"
}

// 응답
{ "status": "ok" }
```

### GET /api/features/{user_id}

```json
// 응답
{
  "user_id": "U1234",
  "recent_clicks": ["item_001", "item_002"],
  "session_interest": { "상의": 3 },
  "click_count": 5
}
```

---

## Redis 데이터 구조

| 키 | 타입 | 설명 | TTL |
|----|------|------|-----|
| `user:{user_id}:recent_clicks` | List | 최근 클릭 상품 ID (최대 20개) | 7일 |
| `user:{user_id}:session_interest` | String (JSON) | 카테고리별 관심 점수 | 7일 |
| `user:{user_id}:click_count` | String (int) | 총 클릭 수 | - |

---

## 실행 방법

### Docker (권장)

```bash
docker-compose up api-gateway redis
```

### 로컬

```bash
pip install -r requirements.txt
REDIS_HOST=localhost uvicorn app:app --port 8000 --reload
```

실행 후 API 문서 확인:

```
http://localhost:8000/docs
```

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `REDIS_HOST` | `redis` | Redis 호스트 |
| `REDIS_PORT` | `6379` | Redis 포트 |
| `SEARCH_ENGINE_URL` | `http://search-engine:8002` | 검색 서비스 URL |
| `REC_MODELS_URL` | `http://rec-models:8003` | 추천 서비스 URL |

---

## 파일 구조

```
api_gateway/
  app.py            # FastAPI 앱, 엔드포인트 정의
  feature_store.py  # Redis 연동 (세션/클릭 이력 관리)
  Dockerfile
  requirements.txt
  README.md
```
