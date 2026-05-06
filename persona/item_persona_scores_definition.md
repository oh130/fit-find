# item_persona_scores definition

## Purpose

`item_persona_scores` is an item-level 9-persona suitability table.

It is used for:

- persona-aware recommendation scoring
- item-side persona lookup
- simulator item sampling logic

## Inputs

- `data/processed/item_master_test.csv`
- `data/processed/item_master.csv`

## Outputs

- test mode: `data/processed/item_persona_scores_test.csv`
- production mode: `data/processed/item_persona_scores.csv`

## Output columns

- `article_id`
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

- scores are heuristic item suitability scores
- ratios are computed so that the 9 persona ratios sum to `1.0`
- `top_persona` is the persona with the highest ratio

## Main feature mapping

- `trendsetter`: trendy item, new item, popular item
- `practical`: basic item, solid appearance, neutral color
- `value`: low-price item, basic item
- `brand_loyal`: stable line or department identity, popularity
- `impulse`: popular item, new item, trendy item
- `careful`: established item, mid/high price, clear category identity
- `repeat_stable`: basic item, solid appearance, stable popularity
- `color_focus`: clear single-color identity, solid appearance
- `category_focus`: clear category hierarchy and product identity
