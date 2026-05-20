# 검색 엔진 실험 리포트

## 실행 정보

- 생성 시각: 2026-05-20T08:46:03.113217+00:00
- 실행 모드: `dev`
- 검색 엔드포인트: `http://127.0.0.1:8002/search`
- 재현 시드: `42`
- 평가 샘플 수: `5`
- 원본 샘플링 상한: `5`

## 데이터 구성 및 분할

- 상품 메타데이터: `data/raw/articles.csv`
- 이미지 데이터: `D:/imagedata/<첫 세 자리>/<article_id>.jpg`
- 검색 인덱스 입력: raw article 메타데이터와 이미지 경로를 매칭한 멀티모달 catalog
- 분할 방식: 시간 기반 `train / valid / test = 8 / 1 / 1`
- split summary: train=0.8, valid=0.1, test=0.1
- test split rows=20000, unique_sessions=2936, range=2025-02-11T18:44:56 ~ 2025-02-16T08:40:44
- 평가셋 구성: 현재 인덱스에 실제로 존재하는 상품을 `color + product type` 기준으로 그룹화해 query bucket을 생성
- relevance 정의: 같은 query bucket에 속한 모든 indexed 상품을 relevant item으로 간주
- 이미지/하이브리드 평가는 query로 사용한 동일 상품을 가능한 경우 점수 계산에서 제외해, exact-image replay 대신 similar-item retrieval을 측정

## 평가 기준

- Offline top-k 기준: `k=10`
- 응답 시간 목표: 평균 API latency `<= 200ms`
- 목표 정확도: `MRR >= 0.55`, `NDCG@10 >= 0.50`
- 검색 평가는 retrieval top-k 기반이므로 별도 negative sampling 없이 전체 searchable universe 대비 측정

## 지표 정의

- `MRR`: 첫 relevant item의 역순위 평균
- `NDCG@10`: 상위 10개 결과의 정규화 누적 이득
- `HitRate@10`: 상위 10개 결과 안에 relevant item이 하나라도 포함될 비율
- `API Latency`: `/search` 응답 본문에 기록된 서버 측 처리 시간
- `Wall Latency`: 평가 스크립트에서 관측한 클라이언트 왕복 시간

## 모달리티별 결과

| 모달리티 | 샘플 수 | MRR | NDCG@10 | HitRate@10 | 평균 API Latency(ms) | P95 API Latency(ms) | 평균 Wall Latency(ms) | 목표 통과 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 텍스트 | 5 | 0.8667 | 0.8243 | 1.0000 | 97.89 | 139.29 | 109.36 | PASS |
| 이미지 | 5 | 0.6500 | 0.5554 | 0.8000 | 488.56 | 593.09 | 534.64 | FAIL |
| 하이브리드 | 5 | 0.6622 | 0.6892 | 1.0000 | 449.39 | 479.73 | 493.50 | FAIL |

## 베이스라인 비교

- 베이스라인: BM25 text-only
- BM25 MRR: `1.0000`
- BM25 NDCG@10: `1.0000`
- CLIP text MRR 개선폭: `-0.1333`
- CLIP text NDCG@10 개선폭: `-0.1757`

## 재현 설정

- CLIP checkpoint: `openai/clip-vit-base-patch32`
- Index: `FAISS IndexHNSWFlat` + cosine-style inner product
- Random seed: `42`
- Dev index는 실제 이미지가 존재하는 상품만 사용

## 쿼리 미리보기

| query_id | query | 대표 상품 id | relevant 수 | source |
| --- | --- | --- | ---: | --- |
| search_q_0001 | dark blue jacket | 0647490002 | 4 | catalog_grouped |
| search_q_0002 | black socks | 0181448105 | 3 | catalog_grouped |
| search_q_0003 | dark blue t-shirt | 0386859053 | 2 | catalog_grouped |
| search_q_0004 | white vest top | 0572799002 | 2 | catalog_grouped |
| search_q_0005 | yellowish brown jacket | 0816363001 | 2 | catalog_grouped |
