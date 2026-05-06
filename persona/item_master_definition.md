# item_master 정의서

## 목적

`item_master`는 상품 단위 기준 테이블이다.

이 테이블은 아래 작업의 공통 입력으로 사용한다.

- 상품별 9개 페르소나 점수 계산
- 사용자 구매 이력과 상품 속성 조인
- 시뮬레이터 상품 샘플링
- 추천/검색 팀의 상품 메타 입력

## 생성 입력

- `data/raw/articles.csv`
- `data/raw/transactions_train.csv`

## 생성 출력

- 테스트 모드: `data/processed/item_master_test.csv`
- 운영 모드: `data/processed/item_master.csv`

## 최종 컬럼

### 원본 기반 컬럼

- `article_id`
- `product_code`
- `prod_name`
- `product_type_name`
- `product_group_name`
- `graphical_appearance_name`
- `colour_group_name`
- `perceived_colour_master_name`
- `department_name`
- `index_name`
- `index_group_name`
- `section_name`
- `garment_group_name`
- `detail_desc`

### 파생 카테고리 컬럼

- `category_l1`
  - 우선순위: `index_group_name -> index_name -> product_group_name`
- `category_l2`
  - 우선순위: `section_name -> department_name -> product_group_name`
- `category_l3`
  - 우선순위: `product_type_name -> product_group_name -> prod_name`

### 거래 집계 컬럼

- `popularity`
  - 해당 상품 구매 건수
- `price_mean`
  - 거래 기준 평균 가격
- `price_bucket`
  - `low`, `mid`, `high`, `unknown`
- `first_purchase_date`
- `last_purchase_date`
- `item_age_days`
  - 데이터셋 마지막 거래일 기준 경과 일수
- `is_new_item`
  - 최근 7일 이내 구매 이력이 있으면 `True`

### 성향 파생 컬럼

- `is_basic`
  - 기본템 성격 여부
- `is_trendy`
  - 트렌드 성격 여부

## 파생 규칙

### price_bucket

- `price_mean` 분포 기준 33%, 66% 분위수로 구간화
- 구간 예시:
  - 하위 33%: `low`
  - 중간 34~66%: `mid`
  - 상위 34%: `high`

### is_basic

아래 조건 중 하나라도 만족하면 `True`로 본다.

- `graphical_appearance_name == Solid`
- `section_name`가 기본 라인 성격
- `garment_group_name`가 `Jersey Basic`, `Trousers`, `Shirts`, `Knitwear` 기본군

### is_trendy

아래 조건 중 하나라도 만족하면 `True`로 본다.

- `section_name`가 `Trend`, `Divided`, `Projects` 계열
- `department_name`가 트렌드 라인 성격

## 활용 예시

### 상품별 페르소나 점수 계산

- 트렌드세터 점수:
  - `is_trendy`, 신상품 여부, 인기 여부 반영
- 실용주의자 점수:
  - `is_basic`, `Solid`, 무채색 여부 반영
- 가성비추구형 점수:
  - `price_bucket == low` 반영
- 색상집중형 점수:
  - `colour_group_name` 반영
- 카테고리집중형 점수:
  - `category_l3` 반영

## 다음 단계

`item_master` 생성 후 바로 이어서 할 작업:

1. `customer_purchase_profile.csv` 생성
2. `user_persona_scores.csv` 생성
3. `item_persona_scores.csv` 생성
