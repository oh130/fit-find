# Simulator

This directory contains the event simulator used to generate synthetic fashion-commerce behavior logs.

## Main Files

- `main.py`
  - simulator entry point
- `persona_config_9.yaml`
  - 9-persona simulator configuration
- `config.yaml`
  - legacy 6-persona configuration kept for reference

## 2026-05-13 Update

The simulator was updated to align with the 9-persona data pipeline.

Completed changes:
- `main.py` now supports `persona_config_9.yaml`
- all 9 personas are treated at the same level
- event flow is aligned to `search / view / cart / purchase`
- a `dry-run` mode was added for local validation without Docker or API dependencies

Supported personas:
- `trendsetter`
- `practical`
- `value`
- `brand_loyal`
- `impulse`
- `careful`
- `repeat_stable`
- `color_focus`
- `category_focus`

## Dry-Run Example

```powershell
$env:SIMULATOR_DRY_RUN="1"
$env:SIMULATOR_MAX_SESSIONS="18"
python simulator/main.py
```

## Docker / API Mode

When `SIMULATOR_DRY_RUN` is not set, the simulator uses the normal API-connected flow and sends events to the running services.

See also:
- `docs/data_simulator_work_report_2026-05-13.md`
- `docs/remaining_issues.txt`
