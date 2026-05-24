"""Path resolution helpers for recommendation serving data."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

_SUPPORTED_DATA_MODES = {"dev", "test", "production"}
_MODE_SUFFIX = {
    "dev": "_dev",
    "test": "_test",
    "production": "",
}


def data_mode() -> str:
    """Return the configured serving data mode."""

    raw_mode = os.getenv("REC_DATA_MODE", "dev").strip().lower()
    if raw_mode == "prod":
        raw_mode = "production"
    return raw_mode if raw_mode in _SUPPORTED_DATA_MODES else "dev"


def configured_path(env_var: str) -> Path | None:
    raw_path = os.getenv(env_var)
    if not raw_path:
        return None
    return Path(raw_path).expanduser()


def mode_order() -> tuple[str, ...]:
    mode = data_mode()
    if mode == "production":
        return ("production", "dev", "test")
    if mode == "test":
        return ("test", "dev", "production")
    return ("dev", "production", "test")


def processed_mode_candidates(stem: str, extension: str) -> tuple[Path, ...]:
    """Return data-mode-aware processed file candidates in preferred order."""

    return tuple(
        PROCESSED_DIR / f"{stem}{_MODE_SUFFIX[mode]}{extension}"
        for mode in mode_order()
    )


def resolve_processed_path(env_var: str, stem: str, extension: str) -> Path:
    """Resolve a required processed artifact path.

    An explicit environment variable is authoritative. Without one, the first
    existing mode-aware candidate is used, falling back to the preferred mode
    path so callers can raise a clear missing-file error.
    """

    explicit_path = configured_path(env_var)
    if explicit_path is not None:
        return explicit_path

    candidates = processed_mode_candidates(stem, extension)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_existing_processed_path(env_var: str, stem: str, extension: str) -> Path | None:
    """Resolve an optional processed artifact path if it exists."""

    explicit_path = configured_path(env_var)
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None

    for path in processed_mode_candidates(stem, extension):
        if path.exists():
            return path
    return None
