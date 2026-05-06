# customer_purchase_profile 정의서

## 목적

`customer_purchase_profile`은 고객 단위 구매 요약 테이블이다.

이 테이블은 아래 작업의 공통 입력으로 사용한다.

- 고객별 9개 페르소나 점수 계산
- 시뮬레이터용 가상 고객 분포 설계
- 추천팀의 사용자 장기 성향 feature 생성
- LLM팀의 사용자 요약 입력 생성

## 생성 입력

- `data/raw/customers.csv`
- `data/raw/transactions_train.csv`
- `data/processed/item_master_test.csv` 또는 `data/processed/item_master.csv`

## 생성 출력

- 테스트 모드: `data/processed/customer_purchase_profile_test.csv`
- 운영 모드: `data/processed/customer_purchase_profile.csv`

## 최종 컬럼

### 고객 기본 컬럼

- `customer_id`
- `age`
- `age_bucket`
- `fashion_news_frequency`
- `club_member_status`

### 구매 집계 컬럼

- `purchase_count`
- `active_days`
- `first_purchase_date`
- `last_purchase_date`
- `recency_days`
- `avg_gap_days`
- `avg_price`
- `same_day_multi_buy_avg`

### 구매 다양성 컬럼

- `unique_items`
- `unique_categories`

### 선호 비중 컬럼

- `top_category`
- `top_category_share`
- `top_color`
- `top_color_share`
- `top_department`
- `top_department_share`

### 성향 파생 컬럼

- `solid_share`
- `basic_share`
- `neutral_color_share`
- `low_price_ratio`
- `repeat_article_share`
- `trendy_item_ratio`

## 파생 규칙

### unique_categories

- `item_master.category_l3` 기준 고유 카테고리 수

### top_category_share

- 가장 많이 구매한 `category_l3` 비율

### top_color_share

- 가장 많이 구매한 `colour_group_name` 비율

### top_department_share

- 가장 많이 구매한 `department_name` 비율

### solid_share

- `graphical_appearance_name == Solid` 인 구매 비율

### basic_share

- `item_master.is_basic == True` 인 구매 비율

### neutral_color_share

- `Black`, `White`, `Grey`, `Beige` 계열 구매 비율

### low_price_ratio

- `item_master.price_bucket == low` 인 구매 비율

### repeat_article_share

- 동일 상품 반복 구매분 비율
- 계산 예시:
  - 총 구매 10건, 고유 상품 7개이면 반복 구매분은 3건
  - `repeat_article_share = 3 / 10 = 0.3`

### same_day_multi_buy_avg

- 구매가 발생한 날짜별 구매 건수의 평균

### avg_gap_days

- 구매가 발생한 날짜들 사이 평균 간격 일수
- 구매일이 1개뿐이면 빈값 또는 0으로 처리 가능

### trendy_item_ratio

- `item_master.is_trendy == True` 인 구매 비율

## 활용 예시

### 트렌드세터 점수

- `recency_days`
- `unique_items`
- `unique_categories`
- `trendy_item_ratio`

### 실용주의자 점수

- `solid_share`
- `basic_share`
- `neutral_color_share`

### 가성비추구형 점수

- `avg_price`
- `low_price_ratio`

### 브랜드충성형 점수

- `top_department_share`

### 충동구매형 점수

- `same_day_multi_buy_avg`

### 신중탐색형 점수

- `avg_gap_days`
- `purchase_count`

## 다음 단계

`customer_purchase_profile` 생성 후 이어서 할 작업:

1. `user_persona_scores.csv` 생성
2. `item_persona_scores.csv` 생성
3. 시뮬레이터용 고객 분포 설계
