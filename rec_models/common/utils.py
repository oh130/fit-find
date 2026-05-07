"""
추천 모델 공용 유틸 함수 모음.

역할:
- seed 고정, device 설정, checkpoint 저장/로드, logging 같은 반복 작업을 모은다.
- 학습 코드와 추론 코드의 보조 기능을 일관된 방식으로 제공한다.

작성 지침:
- 모델 로직과 무관한 반복 작업만 유틸로 둔다.
- 범용성이 낮은 함수는 개별 모듈로 남기고, 여러 곳에서 쓰일 때만 이동한다.
- 부작용이 있는 유틸은 동작 시점과 입출력을 명확히 기록한다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for experiment metadata."""

    return datetime.now(timezone.utc).isoformat()


def write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    """Write one JSON report file, creating parent directories if needed."""

    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_experiment_report(
    *,
    experiment_name: str,
    stage: str,
    data_path: Path,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Wrap metrics with the metadata needed for reproducible experiment logs."""

    return {
        "experiment": {
            "name": experiment_name,
            "stage": stage,
            "generated_at_utc": utc_timestamp(),
            "data_path": str(data_path.expanduser().resolve()),
            "config": config,
        },
        "metrics": metrics,
    }
