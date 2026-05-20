# Search Engine

이 디렉토리는 멀티모달 검색 엔진 서비스 전체를 담당합니다.  
제가 맡은 범위는 크게 4가지입니다.

1. 상품 텍스트/이미지를 CLIP 임베딩으로 변환
2. FAISS HNSW 인덱스로 Top-K 유사 상품 검색
3. FastAPI 기반 검색 API 제공
4. 검색 품질 평가와 리포트 생성

이 README는 이 파트를 처음 보는 팀원이 빠르게 구조를 이해하고, 직접 실행하고, 평가까지 해볼 수 있도록 정리한 문서입니다.

## 이 서비스가 프로젝트에서 하는 일

검색 엔진은 `localhost:8002`에서 동작하는 독립 서비스입니다.

- 입력:
  - 텍스트 쿼리
  - 이미지 쿼리
  - 텍스트 + 이미지 하이브리드 쿼리
- 출력:
  - 유사 상품 Top-K
  - 검색 유형(`text`, `image`, `hybrid`)
  - 유사도 점수
  - 응답 시간

프로젝트 전체 구조에서는 다음과 같이 사용됩니다.

- 프론트엔드 또는 API Gateway가 검색 요청을 보냄
- `search_engine/app.py`가 요청을 받음
- `search_engine/search_engine.py`가 CLIP 임베딩과 벡터 검색 수행
- 결과를 JSON으로 반환

## 현재 구현 상태

현재 구현된 핵심 기능은 다음과 같습니다.

- OpenAI CLIP(`openai/clip-vit-base-patch32`) 기반 텍스트 임베딩
- OpenAI CLIP 기반 이미지 임베딩
- 텍스트 전용 검색
- 이미지 전용 검색
- 텍스트 + 이미지 하이브리드 검색
- FAISS HNSW 기반 최근접 검색
- 검색 인덱스 캐시 저장/로드
- `/search`, `/api/search`, `/health`, `/api/images/{article_id}` API 제공
- 검색 성능 평가 스크립트 제공
- 대시보드에 연결 가능한 JSON/Markdown 리포트 생성

최근에는 이미지 검색 품질을 높이기 위해, 이미지 쿼리에서 단순 이미지 벡터 유사도만 쓰지 않고, 가장 가까운 시각적 anchor 상품의 메타데이터를 질의 힌트로 사용해 재랭킹하는 로직을 추가했습니다.

## 주요 파일 설명

- [app.py](/C:/Users/user/multimodal-search/search_engine/app.py)
  - FastAPI 서버
  - `/search`, `/api/search`, `/health`, `/api/images/{article_id}` 제공
  - 검색 엔진 인스턴스 로딩 및 warm-up 담당

- [search_engine.py](/C:/Users/user/multimodal-search/search_engine/search_engine.py)
  - CLIP 모델 로딩
  - 텍스트/이미지 임베딩
  - 상품 데이터 로딩
  - FAISS HNSW 인덱스 구축
  - 텍스트/이미지/하이브리드 검색 로직
  - 인덱스 캐시 저장/복원

- [query_expansion.py](/C:/Users/user/multimodal-search/search_engine/query_expansion.py)
  - 한국어 패션 질의를 영문 패션 키워드로 보강
  - 예: 색상, 소재, 의류명, 스타일 관련 표현 확장

- [generate_search_metrics_report.py](/C:/Users/user/multimodal-search/search_engine/generate_search_metrics_report.py)
  - 평가셋 생성
  - 실제 검색 API 호출
  - `MRR`, `NDCG@10`, `HitRate@10`, latency 계산
  - JSON/Markdown 리포트 생성

- [evaluate_search_engine.py](/C:/Users/user/multimodal-search/search_engine/evaluate_search_engine.py)
  - 저장된 평가셋 CSV를 기준으로 검색 API를 다시 측정하는 보조 스크립트

- [build_search_pair_manifest.py](/C:/Users/user/multimodal-search/search_engine/build_search_pair_manifest.py)
  - `articles.csv`와 이미지 디렉토리를 `article_id` 기준으로 매칭해 manifest 생성

- [build_search_api_request.py](/C:/Users/user/multimodal-search/search_engine/build_search_api_request.py)
  - 이미지 파일을 base64로 인코딩해 `/api/search` 요청 JSON을 만드는 보조 도구

## 검색 데이터가 만들어지는 방식

검색 엔진은 상품 텍스트 정보와 이미지 정보를 함께 사용합니다.

기본 데이터 소스:

- 텍스트 메타데이터
  - `data/raw/articles.csv`
  - `data/processed/articles_feature.csv`
  - `data/processed/item_master_dev.csv`
  - `data/processed/item_features_dev.csv`

- 이미지 데이터
  - 기본 경로: `D:/imagedata`
  - Docker 내부 기본 경로: `/imagedata`

이미지는 H&M 데이터셋 규칙에 맞춰 아래 형태로 찾습니다.

- 예: `0956217002` 상품 이미지
  - `D:/imagedata/095/0956217002.jpg`

검색 엔진은 `article_id`를 10자리 기준으로 정규화해서 텍스트/이미지/상품 메타데이터를 연결합니다.

## 실행 모드

현재 검색 엔진은 주로 아래 두 모드로 사용합니다.

- `test`
  - 더미 상품과 더미 이미지를 사용
  - 빠른 개발/기본 동작 확인용

- `dev`
  - 실제 전처리 데이터(`data/processed`)와 실제 이미지(`D:/imagedata`) 사용
  - 현재 우리가 주로 검증하는 모드
  - `production`보다 훨씬 가볍고 실험하기 좋음

현재 `docker-compose.yml` 기준 기본 모드는 `dev`입니다.

## 환경 변수

자주 쓰는 환경 변수:

- `SEARCH_ENGINE_MODE`
  - `test`, `dev`, `production`

- `SEARCH_ENGINE_DATA_ROOT`
  - 검색 엔진이 읽을 데이터 루트
  - Docker 기본값: `/app/data/processed`

- `SEARCH_ENGINE_IMAGE_ROOT`
  - 검색 엔진이 읽을 이미지 루트
  - Docker 기본값: `/imagedata`

- `IMAGE_HOST_ROOT`
  - 호스트 이미지 경로
  - 기본값: `D:/imagedata`

- `IMAGE_RUNTIME_ROOT`
  - 컨테이너 내부 이미지 경로
  - 기본값: `/imagedata`

- `SEARCH_ENGINE_DEV_SAMPLE_SIZE`
  - `dev` 모드에서 사용할 샘플 상품 수

## API 규격

포트:

- `8002`

주요 엔드포인트:

- `POST /search`
- `POST /api/search`
- `GET /health`
- `GET /api/images/{article_id}`

### 1. 검색 API

요청 형식:

```json
{
  "query": "black dress",
  "image_base64": null,
  "top_k": 10,
  "use_cache": true
}
```

응답 형식:

```json
{
  "search_type": "text",
  "results": [
    {
      "product_id": "0538226001",
      "name": "Lovely skirt",
      "score": 0.588480711,
      "price": 0,
      "image_url": "/api/images/0538226001",
      "category": "",
      "color": "",
      "product_type": ""
    }
  ],
  "latency_ms": 42.5,
  "total_count": 1
}
```

`search_type`은 다음 중 하나입니다.

- `text`
- `image`
- `hybrid`

### 2. 이미지 조회 API

기본:

- `GET /api/images/{article_id}`
  - 해당 상품 이미지를 binary로 반환

base64가 필요할 때:

- `GET /api/images/{article_id}?format=base64`
  - JSON으로 `image_base64` 반환

예시:

```json
{
  "article_id": "0538226001",
  "normalized_article_id": "0538226001",
  "image_base64": "....",
  "content_type": "image/jpeg"
}
```

## 로컬 실행 방법

### 1. 패키지 설치

```powershell
pip install -r .\search_engine\requirements.txt
```

### 2. dev 모드 실행

```powershell
cd C:\Users\user\multimodal-search\search_engine
$env:SEARCH_ENGINE_MODE="dev"
$env:SEARCH_ENGINE_IMAGE_ROOT="D:\imagedata"
$env:SEARCH_ENGINE_DATA_ROOT="C:\Users\user\multimodal-search\data\processed"
$env:SEARCH_ENGINE_DEV_SAMPLE_SIZE="300"
python .\app.py
```

### 3. 상태 확인

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8002/health -UseBasicParsing | Select-Object -Expand Content
```

## Docker 실행 방법

프로젝트 루트에서:

```powershell
docker-compose up --build
```

주요 주소:

- 검색 엔진: [http://localhost:8002](http://localhost:8002)
- Swagger: [http://localhost:8002/docs](http://localhost:8002/docs)
- API Gateway: [http://localhost:8000](http://localhost:8000)
- 대시보드: [http://localhost:8501](http://localhost:8501)

## 직접 테스트하는 방법

### 텍스트 검색

```powershell
$body = @{
  query = "black dress"
  image_base64 = $null
  top_k = 10
} | ConvertTo-Json

Invoke-WebRequest `
  -Uri http://127.0.0.1:8002/search `
  -Method POST `
  -ContentType "application/json" `
  -Body $body `
  -UseBasicParsing | Select-Object -Expand Content
```

### 이미지 검색

```powershell
$body = @{
  query = ""
  image_base64 = (Get-Content -Path C:\Users\user\multimodal-search\search_engine\0956217002_base64.txt -Raw)
  top_k = 10
} | ConvertTo-Json -Depth 3

Invoke-WebRequest `
  -Uri http://127.0.0.1:8002/search `
  -Method POST `
  -ContentType "application/json" `
  -Body $body `
  -UseBasicParsing | Select-Object -Expand Content
```

### 하이브리드 검색

```powershell
$body = @{
  query = "black dress"
  image_base64 = (Get-Content -Path C:\Users\user\multimodal-search\search_engine\0956217002_base64.txt -Raw)
  top_k = 10
} | ConvertTo-Json -Depth 3

Invoke-WebRequest `
  -Uri http://127.0.0.1:8002/search `
  -Method POST `
  -ContentType "application/json" `
  -Body $body `
  -UseBasicParsing | Select-Object -Expand Content
```

## 검색 성능 평가 방법

평가 스크립트:

- [generate_search_metrics_report.py](/C:/Users/user/multimodal-search/search_engine/generate_search_metrics_report.py)

측정 지표:

- `HitRate@10`
- `MRR`
- `NDCG@10`
- 평균 API latency
- P95 latency

명세 기준 목표:

- `Latency <= 200ms`
- `MRR >= 0.55`
- `NDCG@10 >= 0.50`

### 로컬/호스트에서 실행

```powershell
python .\search_engine\generate_search_metrics_report.py
```

필요하면 endpoint를 명시:

```powershell
python .\search_engine\generate_search_metrics_report.py --endpoint http://127.0.0.1:8002/search
```

생성 파일:

- 평가셋 CSV: [search_eval_set.csv](/C:/Users/user/multimodal-search/evaluation/search_eval_set.csv)
- 평가 JSON: [search_metrics_report.json](/C:/Users/user/multimodal-search/evaluation/search_metrics_report.json)
- 평가 문서: [search_experiments.md](/C:/Users/user/multimodal-search/docs/search_experiments.md)

### 현재 평가 방식 요약

평가는 실제 `/search` API를 호출하는 방식입니다.

- `text`: 텍스트만 전달
- `image`: 이미지 base64만 전달
- `hybrid`: 텍스트 + 이미지 전달

`dev` 모드에서는 주로:

- 전처리된 상품군에서 평가셋 생성
- query별 relevant item 집합 생성
- 응답의 ranked list와 relevant set을 비교

또한 현재 평가는 query-side cache를 끄고(`use_cache=false`) 측정해서, 캐시 히트로 latency가 과도하게 낮게 나오는 문제를 줄였습니다.

## 최근 작업에서 바뀐 핵심 내용

이번까지 반영된 중요한 변경 사항은 아래와 같습니다.

- `data_embedding.py` 의존 제거
- `search_engine.py` self-contained 구조로 정리
- CLIP 텍스트/이미지 임베딩 내부 구현
- 이미지 전용 검색 품질 개선
  - 이미지 쿼리에서 가장 가까운 시각적 상품(anchor)을 찾음
  - anchor 상품의 색상/상품유형 메타데이터를 질의 힌트로 사용
  - 이미지 유사도 + 멀티모달 유사도 + 텍스트 힌트 점수를 조합해 재랭킹
- `/api/images/{article_id}?format=base64` 지원
- Docker 환경에서도 이미지 경로를 ENV로 바꿀 수 있게 수정
- 평가 스크립트가 컨테이너 내부 경로에서도 동작하도록 보강
- 평가 스크립트에 metric fallback 구현

## 현재 확인된 결과

최근 직접 확인한 결과는 아래와 같습니다.

- 텍스트 검색은 목표치를 안정적으로 통과
- 하이브리드 검색도 목표치를 통과
- 이미지 검색은 기존보다 품질이 올라갔고, 최신 로직 기준 로컬 검증 수치는:
  - `MRR = 0.6839`
  - `NDCG@10 = 0.6748`

즉 이미지 검색도 목표치인

- `MRR >= 0.55`
- `NDCG@10 >= 0.50`

를 넘는 방향으로 개선된 상태입니다.

다만 Docker 호스트에서 `localhost:8002`로 붙는 경로는 환경에 따라 간헐적으로 빈 응답이 섞일 수 있어서, 최종 리포트 재생성 시에는 로컬 실행과 컨테이너 내부 실행 둘 다 확인하는 편이 안전합니다.

## 팀원이 보면 좋은 체크포인트

이 파트를 이어서 볼 때는 아래 순서로 보면 가장 빠릅니다.

1. [app.py](/C:/Users/user/multimodal-search/search_engine/app.py) 에서 API 입구 확인
2. [search_engine.py](/C:/Users/user/multimodal-search/search_engine/search_engine.py) 에서
   - 데이터 로딩
   - 인덱스 구축
   - 검색 로직
   - 이미지 전용 재랭킹
3. [generate_search_metrics_report.py](/C:/Users/user/multimodal-search/search_engine/generate_search_metrics_report.py) 에서 평가 방식 확인

특히 이미지 검색 품질을 더 건드릴 일이 있다면 [search_engine.py](/C:/Users/user/multimodal-search/search_engine/search_engine.py) 의 `_search_image_mode()`부터 보는 것이 가장 좋습니다.

## 참고

- 전체 시스템 구조: [README.md](/C:/Users/user/multimodal-search/README.md)
- 검색 품질 리포트: [search_experiments.md](/C:/Users/user/multimodal-search/docs/search_experiments.md)
- 대시보드 소스: [streamlit_app.py](/C:/Users/user/multimodal-search/evaluation/streamlit_app.py)
