# 데이터/시뮬레이터 작업 보고서
작성일: 2026-05-13

## 1. 작업 목적

본 작업의 목적은 H&M 원본 데이터를 기반으로 추천 시스템과 시뮬레이터가 공통으로 사용할 수 있는 데이터 파이프라인을 정리하고, 9개 페르소나 기반 행동 로그가 실제로 생성되는 구조를 안정화하는 데 있었다.

특히 아래 두 문제를 우선 해결 대상으로 설정하였다.

1. 시뮬레이터가 6개 페르소나만 사용하고 있어 9개 페르소나 체계와 불일치하는 문제
2. 학습 데이터에서 일부 페르소나만 과도하게 등장하고 나머지 페르소나는 거의 생성되지 않는 문제

## 2. 수행 내용

### 2.1 H&M 데이터 구조 분석 및 전처리 파이프라인 정리

사용 데이터셋:
- `articles.csv`
- `customers.csv`
- `transactions_train.csv`

주요 작업:
- 상품/고객/거래 컬럼 의미 분석
- 추천/검색에 필요한 핵심 컬럼 선별
- pandas 기반 전처리 파이프라인 정리
- `merge`, `groupby`, `aggregation`, 결측치 처리, 파생 컬럼 생성 로직 정리

### 2.2 추천 및 시뮬레이터용 중간 산출물 생성

구축한 산출물:
- `item_master`
- `customer_purchase_profile`
- `user_persona_scores`
- `item_persona_scores`
- `sim_users`
- `simulated_events`
- `train / valid / test` 이벤트 분할

핵심 의미:
- 원본 데이터를 바로 쓰지 않고, 추천 모델과 시뮬레이터가 사용할 수 있는 형태로 feature와 로그 구조를 가공함

### 2.3 9개 페르소나 구조 정리

최종 사용 페르소나:
- `trendsetter`
- `practical`
- `value`
- `brand_loyal`
- `impulse`
- `careful`
- `repeat_stable`
- `color_focus`
- `category_focus`

관련 문서:
- `persona/persona_definition.md`
- `persona/user_persona_scores_definition.md`
- `persona/item_persona_scores_definition.md`

### 2.4 시뮬레이터 9개 페르소나 정합성 수정

기존 상태:
- `simulator/main.py`는 6개 페르소나용 `config.yaml` 기준
- 데이터 파이프라인은 9개 페르소나용 `persona_config_9.yaml` 기준

수정 내용:
- `simulator/main.py`를 `persona_config_9.yaml` 기준으로 수정
- `preferred_categories`, `preferred_colors`, `search_prob`, `view_prob`, `cart_prob`, `purchase_prob`, `session_searches` 스키마 지원
- `search / view / cart / purchase` 이벤트 흐름 기준으로 정리
- Docker 없이 로컬 검증 가능한 `dry-run` 모드 추가

결과:
- 9개 페르소나 모두 실시간 이벤트 생성 가능 상태로 전환

### 2.5 페르소나 분포 불균형 보정

수정 파일:
- `data_pipeline/build_user_persona_scores.py`
- `data_pipeline/build_item_persona_scores.py`
- `simulator/persona_config_9.yaml`

보정 방향:
- `value`, `careful` 쏠림 완화
- `practical`, `brand_loyal`, `repeat_stable`, `color_focus`, `category_focus` 활성화
- 0구매 고객 제거
- 분위수 기반 점수 반영으로 극단적 편향 완화

## 3. 검증 결과

### 3.1 test 모드

핵심 결과:
- 9개 페르소나 모두 이벤트 생성 확인
- `missing_search_query_rows = 0`
- `missing_item_rows = 0`

### 3.2 dev 모드

검증 파일:
- `data/processed/simulated_events_validation_dev.json`

요약:
- 총 이벤트: `200,000`
- 유저 수: `9,444`
- 세션 수: `29,313`

active_persona 분포:
- `careful`: 35,049
- `value`: 28,736
- `practical`: 27,704
- `impulse`: 21,569
- `brand_loyal`: 20,332
- `color_focus`: 18,317
- `repeat_stable`: 18,248
- `trendsetter`: 16,555
- `category_focus`: 13,490

event_type 분포:
- `view`: 124,015
- `cart`: 37,547
- `search`: 30,050
- `purchase`: 8,388

해석:
- 9개 페르소나가 모두 실제 로그에 의미 있는 규모로 등장함
- 특정 몇 개 페르소나만 사실상 죽어 있던 초기 상태는 해소됨
- 시연 및 학습용 데이터로 사용할 수 있는 수준으로 안정화됨

## 4. 현재 상태

완료된 범위:
- 9개 페르소나 정의 정리
- 데이터 전처리 파이프라인 정리
- 유저/아이템 페르소나 점수 생성
- 시뮬레이션 유저 생성
- 시뮬레이션 이벤트 로그 생성
- 이벤트 검증 JSON 생성
- train / valid / test 분할 생성
- 시뮬레이터 9페르소나 정합성 수정
- 페르소나 불균형 보정 및 dev 모드 검증 완료

## 5. 결론

이번 작업을 통해 데이터 파이프라인과 시뮬레이터 간 페르소나 정의를 일치시켰고, 9개 페르소나가 모두 실제 이벤트 로그에 반영되는 구조를 확보하였다.

또한 기존에 심각했던 페르소나 분포 불균형 문제를 완화하여, 현재는 비율 기반 추천 로직과 오프라인 시뮬레이션 검증에 투입 가능한 상태로 판단된다.
