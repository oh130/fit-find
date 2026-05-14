# Data Directory

This directory stores local raw data and generated processed outputs.

## Raw Data

Keep the following source files in `data/raw/` locally only:
- `articles.csv`
- `customers.csv`
- `transactions_train.csv`

These files must not be committed to Git.

## Processed Outputs

The pipeline writes generated artifacts into `data/processed/`.

Important generated files include:
- `item_master_*.csv`
- `customer_purchase_profile_*.csv`
- `user_persona_scores_*.csv`
- `item_persona_scores_*.csv`
- `sim_users_*.csv`
- `simulated_events_*.csv`
- `simulated_events_validation_*.json`
- `train_events_*.csv`
- `valid_events_*.csv`
- `test_events_*.csv`
- `event_split_summary_*.json`

## 2026-05-13 Update

As of 2026-05-13:
- the data pipeline and simulator both use the same 9-persona schema
- persona imbalance was rebalanced for both user-side and item-side scoring
- `test` and `dev` mode validations confirmed that all 9 personas appear in generated event logs

See also:
- `docs/data_simulator_work_report_2026-05-13.md`
- `docs/remaining_issues.txt`
- `persona/`
