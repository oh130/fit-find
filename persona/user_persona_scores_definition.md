# user_persona_scores definition

## Purpose

`user_persona_scores` is a customer-level 9-persona ratio table.

It is generated from `customer_purchase_profile` and used for:

- user-side persona ratio lookup
- simulator user initialization
- LLM context summary
- recommendation feature inputs

## Inputs

- `data/processed/customer_purchase_profile_test.csv`
- `data/processed/customer_purchase_profile.csv`

## Outputs

- test mode: `data/processed/user_persona_scores_test.csv`
- production mode: `data/processed/user_persona_scores.csv`

## Output columns

- `customer_id`
- `trendsetter_score`
- `practical_score`
- `value_score`
- `brand_loyal_score`
- `impulse_score`
- `careful_score`
- `repeat_stable_score`
- `color_focus_score`
- `category_focus_score`
- `trendsetter_ratio`
- `practical_ratio`
- `value_ratio`
- `brand_loyal_ratio`
- `impulse_ratio`
- `careful_ratio`
- `repeat_stable_ratio`
- `color_focus_ratio`
- `category_focus_ratio`
- `top_persona`
- `top_persona_ratio`

## Scoring policy

- scores are computed from normalized customer purchase features
- ratios are computed so that the 9 persona ratios sum to `1.0`
- `top_persona` is the persona with the highest ratio

## Main feature mapping

- `trendsetter`: low recency, many unique items, many unique categories, high trendy ratio
- `practical`: high solid share, high basic share, high neutral color share
- `value`: low average price, high low price ratio
- `brand_loyal`: high top department share
- `impulse`: high same-day multi-buy average, higher purchase density
- `careful`: high average purchase gap, lower purchase count
- `repeat_stable`: high repeat article share, practical tendency
- `color_focus`: high top color share
- `category_focus`: high top category share
