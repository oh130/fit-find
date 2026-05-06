# 9개 페르소나 정의서

## 목적

이 문서는 패션 추천 시스템에서 사용할 9개 페르소나를 동일한 레벨의 성향 축으로 정의한다.

활용 목적은 아래와 같다.

- 고객 구매 이력 기반 `user_persona_scores` 생성
- 상품 속성 기반 `item_persona_scores` 생성
- LLM이 자연어 요청을 9개 페르소나 비율로 변환할 때 기준 제공
- 이후 시뮬레이터에서 `search / view / cart / purchase` 행동 확률 설계에 활용

## 기본 원칙

- 총 페르소나는 9개이며, 모두 같은 레벨의 성향으로 취급한다.
- 한 고객은 9개 페르소나에 대해 모두 점수를 가질 수 있다.
- 최종 출력은 `합계 100%`의 비율 벡터로 사용한다.
- 고정 임계값보다 분위수 기반 기준을 우선 사용한다.

## 사용 데이터

### 상품 기준

- `article_id`
- `product_type_name`
- `product_group_name`
- `graphical_appearance_name`
- `colour_group_name`
- `department_name`
- `section_name`
- `garment_group_name`

### 거래 기준

- `customer_id`
- `article_id`
- `t_dat`
- `price`

## 고객 단위 핵심 피처

아래 피처를 고객 단위로 계산한 뒤 페르소나 점수 산출에 사용한다.

- `purchase_count`: 총 구매 횟수
- `active_days`: 구매가 발생한 날짜 수
- `recency_days`: 마지막 구매 이후 경과 일수
- `avg_price`: 평균 구매 가격
- `unique_items`: 고유 상품 수
- `unique_categories`: 고유 카테고리 수
- `top_category_share`: 최다 구매 카테고리 비중
- `top_color_share`: 최다 구매 색상 비중
- `top_department_share`: 최다 구매 department 비중
- `solid_share`: `graphical_appearance_name == Solid` 비중
- `basic_share`: 기본템 계열 상품 비중
- `neutral_color_share`: `Black`, `White`, `Grey`, `Beige` 비중
- `low_price_ratio`: 저가 구간 상품 비중
- `repeat_article_share`: 동일 `article_id` 재구매 비중
- `same_day_multi_buy_avg`: 하루 평균 다건 구매 수
- `avg_gap_days`: 구매 간 평균 일수
- `trendy_item_ratio`: 트렌드 계열 상품 비중

## 9개 페르소나 정의

### 1. 트렌드세터

- 설명:
  - 최근에도 활발히 구매하고, 다양한 상품을 탐색하며, 트렌드 계열 상품 비중이 높은 고객
- 주요 컬럼:
  - `t_dat`, `article_id`, `section_name`, `product_type_name`
- 핵심 피처:
  - `recency_days`, `unique_items`, `unique_categories`, `trendy_item_ratio`
- 판별 기준 예시:
  - `unique_items` 상위 20%
  - `recency_days` 하위 30%
  - `trendy_item_ratio` 상위 30%
- 점수식 예시:
  - `0.4 * unique_items_norm + 0.3 * (1 - recency_norm) + 0.3 * trendy_item_ratio`
- 선호 속성:
  - 트렌드 섹션, 신상품, 다양한 카테고리

### 2. 실용주의자

- 설명:
  - 기본템과 무난한 스타일을 선호하고, 재활용 가능한 아이템 비중이 높은 고객
- 주요 컬럼:
  - `product_type_name`, `graphical_appearance_name`, `colour_group_name`, `garment_group_name`
- 핵심 피처:
  - `solid_share`, `basic_share`, `neutral_color_share`
- 판별 기준 예시:
  - `solid_share` 높음
  - `basic_share` 상위 30%
  - `neutral_color_share` 상위 30%
- 점수식 예시:
  - `0.4 * solid_share + 0.35 * basic_share + 0.25 * neutral_color_share`
- 선호 속성:
  - 기본 상의, 무채색, 심플한 패턴

### 3. 가성비추구형

- 설명:
  - 가격 민감도가 높고, 상대적으로 저렴한 상품을 중심으로 소비하는 고객
- 주요 컬럼:
  - `price`, `product_type_name`
- 핵심 피처:
  - `avg_price`, `low_price_ratio`
- 판별 기준 예시:
  - `avg_price` 하위 30%
  - `low_price_ratio` 상위 30%
- 점수식 예시:
  - `0.5 * (1 - avg_price_norm) + 0.5 * low_price_ratio`
- 선호 속성:
  - 저가 상품, 가격 효율 높은 기본 카테고리

### 4. 브랜드충성형

- 설명:
  - 특정 브랜드 또는 라인 성격의 상품군에 집중하는 고객
- 주요 컬럼:
  - `department_name`, `section_name`
- 핵심 피처:
  - `top_department_share`
- 판별 기준 예시:
  - `top_department_share > 0.8`
- 점수식 예시:
  - `top_department_share`
- 선호 속성:
  - 특정 department 또는 section 반복 소비

주의:

- H&M 데이터에서 `department_name`은 완전한 브랜드가 아닐 수 있으므로, 실제 구현 시 `브랜드/라인 충성형`으로 해석한다.

### 5. 충동구매형

- 설명:
  - 짧은 시간에 여러 상품을 사고, 구매 결정이 빠른 고객
- 주요 컬럼:
  - `t_dat`, `article_id`
- 핵심 피처:
  - `same_day_multi_buy_avg`, `purchase_count`
- 판별 기준 예시:
  - `same_day_multi_buy_avg` 상위 20%
- 점수식 예시:
  - `same_day_multi_buy_avg_norm`
- 선호 속성:
  - 인기 상품, 즉시 반응 가능한 상품군

### 6. 신중탐색형

- 설명:
  - 구매 간격이 길고 구매 빈도는 낮지만, 비교와 탐색 성향이 강한 고객
- 주요 컬럼:
  - `t_dat`, `article_id`
- 핵심 피처:
  - `avg_gap_days`, `purchase_count`, `recency_days`
- 판별 기준 예시:
  - `avg_gap_days` 상위 30%
  - `purchase_count` 하위 30%
- 점수식 예시:
  - `0.6 * avg_gap_days_norm + 0.4 * (1 - purchase_count_norm)`
- 선호 속성:
  - 비교 가능한 상품, 구매 결정이 느린 카테고리

### 7. 재구매안정형

- 설명:
  - 동일 상품 또는 매우 유사한 상품을 반복 구매하는 안정형 고객
- 주요 컬럼:
  - `article_id`
- 핵심 피처:
  - `repeat_article_share`
- 판별 기준 예시:
  - `repeat_article_share > 0.4`
- 점수식 예시:
  - `repeat_article_share`
- 선호 속성:
  - 익숙한 상품, 유사 상품, 반복 구매 가능한 기본템

### 8. 색상집중형

- 설명:
  - 특정 색상 계열에 강한 선호를 보이는 고객
- 주요 컬럼:
  - `colour_group_name`
- 핵심 피처:
  - `top_color_share`
- 판별 기준 예시:
  - `top_color_share > 0.7`
- 점수식 예시:
  - `top_color_share`
- 선호 속성:
  - 동일 색상군, 유사 톤 상품

### 9. 카테고리집중형

- 설명:
  - 특정 상품 카테고리 중심으로 소비가 편중된 고객
- 주요 컬럼:
  - `product_type_name`, `section_name`
- 핵심 피처:
  - `top_category_share`
- 판별 기준 예시:
  - `top_category_share > 0.8`
- 점수식 예시:
  - `top_category_share`
- 선호 속성:
  - 특정 카테고리 내 깊은 탐색과 반복 구매

## 최종 산출 방식

고객마다 아래 9개 점수를 계산한다.

- `trendsetter_score`
- `practical_score`
- `value_score`
- `brand_loyal_score`
- `impulse_score`
- `careful_score`
- `repeat_stable_score`
- `color_focus_score`
- `category_focus_score`

권장 후처리:

- 각 점수를 0 이상 값으로 정규화한다.
- 9개 점수의 합이 1 또는 100이 되도록 변환한다.
- 최종 `user_persona_scores`는 비율 벡터 형태로 저장한다.

예시:

```json
{
  "customer_id": "U1234",
  "trendsetter_score": 0.12,
  "practical_score": 0.22,
  "value_score": 0.18,
  "brand_loyal_score": 0.07,
  "impulse_score": 0.09,
  "careful_score": 0.10,
  "repeat_stable_score": 0.08,
  "color_focus_score": 0.06,
  "category_focus_score": 0.08
}
```

## 다음 단계

이 문서 다음 작업 순서는 아래와 같다.

1. `item_master.csv` 생성
2. `customer_purchase_profile.csv` 생성
3. `user_persona_scores.csv` 생성
4. `item_persona_scores.csv` 생성
5. 시뮬레이터 설정 파일 작성
6. `search / view / cart / purchase` 로그 생성
