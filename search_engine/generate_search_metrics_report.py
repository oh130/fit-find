from __future__ import annotations

import argparse
import base64
import http.client
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_ROOT = SCRIPT_DIR.parent
if (PARENT_ROOT / "evaluation").exists() or (PARENT_ROOT / "docs").exists():
    REPO_ROOT = PARENT_ROOT
else:
    REPO_ROOT = SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from evaluation.metrics import mean_hit_rate_at_k, mean_ndcg_at_k, mean_reciprocal_rank
except ModuleNotFoundError:
    def mean_hit_rate_at_k(ranked_lists, relevant_lists, k):
        hits = []
        for ranked, relevant in zip(ranked_lists, relevant_lists):
            relevant_set = set(relevant)
            hits.append(1.0 if any(item in relevant_set for item in ranked[:k]) else 0.0)
        return float(sum(hits) / len(hits)) if hits else 0.0

    def mean_reciprocal_rank(ranked_lists, relevant_lists):
        rr_scores = []
        for ranked, relevant in zip(ranked_lists, relevant_lists):
            relevant_set = set(relevant)
            reciprocal = 0.0
            for rank, item in enumerate(ranked, start=1):
                if item in relevant_set:
                    reciprocal = 1.0 / rank
                    break
            rr_scores.append(reciprocal)
        return float(sum(rr_scores) / len(rr_scores)) if rr_scores else 0.0

    def mean_ndcg_at_k(ranked_lists, relevant_lists, k):
        ndcg_scores = []
        for ranked, relevant in zip(ranked_lists, relevant_lists):
            relevant_set = set(relevant)
            dcg = 0.0
            for rank, item in enumerate(ranked[:k], start=1):
                if item in relevant_set:
                    dcg += 1.0 / math.log2(rank + 1)
            ideal_hits = min(len(relevant_set), k)
            if ideal_hits <= 0:
                ndcg_scores.append(0.0)
                continue
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcg_scores.append(dcg / idcg if idcg else 0.0)
        return float(sum(ndcg_scores) / len(ndcg_scores)) if ndcg_scores else 0.0

DEFAULT_ENDPOINT = "http://127.0.0.1:8002/search"
DEFAULT_MODE = "auto"
DEFAULT_TOP_K = 10
DEFAULT_METRIC_K = 10
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_MIN_RELEVANT = 2
DEFAULT_TIMEOUT = 60.0
DEFAULT_RANDOM_SEED = 42
SEARCH_DOC_REPORT_PATH = REPO_ROOT / "docs" / "search_experiments.md"
SEARCH_JSON_REPORT_PATH = REPO_ROOT / "evaluation" / "search_metrics_report.json"
SEARCH_EVAL_SET_PATH = REPO_ROOT / "evaluation" / "search_eval_set.csv"
DEV_META_PATH = REPO_ROOT / "data" / "faiss_index" / "search_dev_v2_metadata.json"
TEST_META_PATH = REPO_ROOT / "data" / "faiss_index" / "search_test_v2_metadata.json"
PROD_META_PATH = REPO_ROOT / "data" / "faiss_index" / "search_v2_metadata.json"
DEV_TEST_EVENTS_PATH = REPO_ROOT / "data" / "processed" / "test_events_dev.csv"
DEV_SPLIT_SUMMARY_PATH = REPO_ROOT / "data" / "processed" / "event_split_summary_dev.json"
_LOCAL_ENGINE_CACHE: dict[str, Any] = {}
REQUEST_RETRIES = 3
REQUEST_RETRY_SLEEP_SEC = 1.0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_items_from_metadata(meta_path: Path) -> pd.DataFrame:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        metadata = dict(item.get("metadata", {}))
        product_id = _normalize_article_id(item.get("product_id", ""))
        rows.append(
            {
                "product_id": product_id,
                "name": str(item.get("name", "")).strip(),
                "description": str(item.get("description", "")).strip(),
                "price": _coerce_float(item.get("price", 0.0)),
                "image_path": str(item.get("image_path", "")).strip(),
                "product_type_name": str(metadata.get("product_type_name", "")).strip(),
                "colour_group_name": str(metadata.get("colour_group_name", "")).strip(),
                "department_name": str(metadata.get("department_name", "")).strip(),
                "section_name": str(metadata.get("section_name", "")).strip(),
                "detail_desc": str(metadata.get("detail_desc", "")).strip(),
                "mode": str(metadata.get("mode", payload.get("mode", ""))).strip(),
                "image_available": bool(str(item.get("image_path", "")).strip()),
            }
        )
    return pd.DataFrame(rows)


def _load_items_from_engine(mode: str) -> pd.DataFrame:
    from search_engine import MultimodalSearchEngine

    engine = MultimodalSearchEngine(mode=mode)
    rows: list[dict[str, Any]] = []
    for item in engine.items:
        metadata = dict(item.metadata or {})
        product_id = _normalize_article_id(item.product_id)
        rows.append(
            {
                "product_id": product_id,
                "name": str(item.name).strip(),
                "description": str(item.description).strip(),
                "price": _coerce_float(item.price, 0.0),
                "image_path": str(item.image_path or "").strip(),
                "product_type_name": str(metadata.get("product_type_name", "")).strip(),
                "colour_group_name": str(metadata.get("colour_group_name", "")).strip(),
                "department_name": str(metadata.get("department_name", "")).strip(),
                "section_name": str(metadata.get("section_name", "")).strip(),
                "detail_desc": str(metadata.get("detail_desc", "")).strip(),
                "mode": str(metadata.get("mode", mode)).strip(),
                "image_available": bool(item.image_path or item.image is not None),
            }
        )
    return pd.DataFrame(rows)


def _metadata_path_for_mode(mode: str) -> Path:
    if mode == "test":
        return TEST_META_PATH
    if mode == "dev":
        return DEV_META_PATH
    return PROD_META_PATH


def _load_index_items(mode: str) -> pd.DataFrame:
    meta_path = _metadata_path_for_mode(mode)
    if meta_path.exists():
        return _load_items_from_metadata(meta_path)
    return _load_items_from_engine(mode)


def _get_local_engine(mode: str):
    engine = _LOCAL_ENGINE_CACHE.get(mode)
    if engine is not None:
        return engine
    from search_engine import MultimodalSearchEngine

    engine = MultimodalSearchEngine(mode=mode)
    _LOCAL_ENGINE_CACHE[mode] = engine
    return engine


def _build_query(row: pd.Series) -> str:
    color = str(row.get("colour_group_name", "")).strip()
    product_type = str(row.get("product_type_name", "")).strip()
    name = str(row.get("name", "")).strip()
    if color and product_type:
        return f"{color.lower()} {product_type.lower()}".strip()
    if name:
        return name.strip().lower()
    return str(row.get("description", "")).strip().lower()


def _normalize_article_id(value: object) -> str:
    raw = str(value or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return raw
    return digits[-10:].zfill(10)


def _build_eval_set_from_test_events(
    mode: str,
    items: pd.DataFrame,
    sample_size: int,
    random_seed: int,
) -> pd.DataFrame:
    if mode != "dev" or not DEV_TEST_EVENTS_PATH.exists():
        return pd.DataFrame()

    event_df = pd.read_csv(
        DEV_TEST_EVENTS_PATH,
        dtype={
            "event_type": str,
            "query_text": str,
            "article_id": str,
            "session_id": str,
            "timestamp": str,
        },
    ).fillna("")
    if event_df.empty:
        return pd.DataFrame()

    item_lookup = items.copy()
    item_lookup["product_id"] = item_lookup["product_id"].astype(str).map(_normalize_article_id)
    item_lookup = item_lookup[item_lookup["product_id"].str.strip() != ""].drop_duplicates(subset=["product_id"])
    item_map = item_lookup.set_index("product_id").to_dict("index")

    event_df["article_id"] = event_df["article_id"].astype(str).map(_normalize_article_id)
    event_df = event_df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    grouped = event_df.groupby("session_id", sort=False)
    for _, session in grouped:
        records = session.to_dict("records")
        for idx, record in enumerate(records):
            if str(record.get("event_type", "")).strip().lower() != "search":
                continue
            query = str(record.get("query_text", "")).strip()
            if not query:
                continue

            relevant_items: list[str] = []
            seen: set[str] = set()
            for next_record in records[idx + 1 :]:
                next_type = str(next_record.get("event_type", "")).strip().lower()
                if next_type == "search":
                    break
                if next_type not in {"view", "cart", "purchase"}:
                    continue
                product_id = _normalize_article_id(next_record.get("article_id", ""))
                if not product_id or product_id not in item_map or product_id in seen:
                    continue
                seen.add(product_id)
                relevant_items.append(product_id)

            if not relevant_items:
                continue

            representative_id = next(
                (
                    product_id
                    for product_id in relevant_items
                    if str(item_map.get(product_id, {}).get("image_path", "")).strip()
                ),
                relevant_items[0],
            )
            representative = item_map.get(representative_id, {})
            rows.append(
                {
                    "query": query,
                    "relevant_items": relevant_items,
                    "relevant_count": len(relevant_items),
                    "query_image_path": str(representative.get("image_path", "")).strip(),
                    "query_product_id": representative_id,
                    "source_split": "test_events_dev",
                }
            )

    if not rows:
        return pd.DataFrame()

    eval_df = pd.DataFrame(rows).drop_duplicates(subset=["query", "query_product_id"])
    eval_df = eval_df.sample(
        n=min(sample_size, len(eval_df)),
        random_state=random_seed,
    ).reset_index(drop=True)
    eval_df.insert(0, "query_id", [f"search_q_{idx + 1:04d}" for idx in range(len(eval_df))])
    return eval_df


def _build_eval_set_from_catalog_groups(
    items: pd.DataFrame,
    sample_size: int,
    min_relevant: int,
    random_seed: int,
) -> pd.DataFrame:
    catalog_items = items.copy()
    catalog_items["query"] = catalog_items.apply(_build_query, axis=1)
    catalog_items["product_id"] = catalog_items["product_id"].astype(str).map(_normalize_article_id)
    catalog_items = catalog_items[catalog_items["query"].str.strip() != ""].copy()
    catalog_items = catalog_items[catalog_items["product_id"].str.strip() != ""].copy()
    if "image_available" not in catalog_items.columns:
        catalog_items["image_available"] = False
    catalog_items["has_image"] = (
        catalog_items["image_available"].astype(bool)
        | catalog_items["image_path"].astype(str).str.strip().ne("")
    )

    grouped_rows: list[dict[str, Any]] = []
    for query, group in catalog_items.groupby("query", sort=False):
        relevant_ids = sorted({str(value).strip() for value in group["product_id"] if str(value).strip()})
        if len(relevant_ids) < min_relevant:
            continue

        image_group = group[group["has_image"]].copy()
        if image_group.empty:
            continue

        representative = image_group.sort_values(["product_id"]).iloc[0]
        grouped_rows.append(
            {
                "query": query,
                "relevant_items": relevant_ids,
                "relevant_count": len(relevant_ids),
                "query_image_path": str(representative["image_path"]).strip(),
                "query_product_id": _normalize_article_id(representative["product_id"]),
                "source_split": "catalog_grouped",
            }
        )

    if not grouped_rows:
        return pd.DataFrame()

    eval_df = pd.DataFrame(grouped_rows)
    eval_df = eval_df.sample(
        n=min(sample_size, len(eval_df)),
        random_state=random_seed,
    ).sort_values(["relevant_count", "query"], ascending=[False, True]).reset_index(drop=True)
    eval_df.insert(0, "query_id", [f"search_q_{idx + 1:04d}" for idx in range(len(eval_df))])
    return eval_df


def build_eval_set(
    mode: str,
    sample_size: int,
    min_relevant: int,
    random_seed: int,
) -> pd.DataFrame:
    items = _load_index_items(mode).fillna("")
    if items.empty:
        raise ValueError(f"No indexed items found for mode={mode}")

    catalog_eval_df = _build_eval_set_from_catalog_groups(
        items=items,
        sample_size=sample_size,
        min_relevant=min_relevant,
        random_seed=random_seed,
    )
    if not catalog_eval_df.empty:
        return catalog_eval_df

    event_eval_df = _build_eval_set_from_test_events(
        mode=mode,
        items=items,
        sample_size=sample_size,
        random_seed=random_seed,
    )
    if not event_eval_df.empty:
        return event_eval_df

    items["query"] = items.apply(_build_query, axis=1)
    items = items[items["query"].str.strip() != ""].copy()
    items = items[items["product_id"].str.strip() != ""].copy()
    if "image_available" not in items.columns:
        items["image_available"] = False
    items["has_image"] = (
        items["image_available"].astype(bool)
        | items["image_path"].astype(str).str.strip().ne("")
    )

    grouped_rows: list[dict[str, Any]] = []
    for query, group in items.groupby("query", sort=False):
        relevant_ids = sorted({str(value).strip() for value in group["product_id"] if str(value).strip()})
        if len(relevant_ids) < min_relevant:
            continue

        image_group = group[group["has_image"]].copy()
        if image_group.empty:
            continue

        representative = image_group.sort_values(["product_id"]).iloc[0]
        grouped_rows.append(
            {
                "query": query,
                "relevant_items": relevant_ids,
                "relevant_count": len(relevant_ids),
                "query_image_path": str(representative["image_path"]).strip(),
                "query_product_id": str(representative["product_id"]).strip(),
            }
        )

    eval_df = pd.DataFrame(grouped_rows)
    if eval_df.empty:
        # test 모드 더미 데이터처럼 동일 query 그룹이 충분하지 않은 경우에는
        # 이미지가 있는 각 상품을 자기 자신 relevance 1개짜리 쿼리로 평가한다.
        fallback_items = items[items["has_image"]].copy()
        fallback_rows: list[dict[str, Any]] = []
        for row in fallback_items.itertuples(index=False):
            product_id = str(getattr(row, "product_id", "")).strip()
            query = str(getattr(row, "query", "")).strip()
            if not product_id or not query:
                continue
            fallback_rows.append(
                {
                    "query": query,
                    "relevant_items": [product_id],
                    "relevant_count": 1,
                    "query_image_path": str(getattr(row, "image_path", "") or "").strip(),
                    "query_product_id": product_id,
                    "source_split": "catalog_fallback",
                }
            )
        eval_df = pd.DataFrame(fallback_rows)

    if eval_df.empty:
        raise ValueError(
            "No multimodal evaluation groups were created. "
            "Check image availability or SEARCH_ENGINE_IMAGE_ROOT / /api/images support."
        )

    eval_df = eval_df.sample(
        n=min(sample_size, len(eval_df)),
        random_state=random_seed,
    ).sort_values(["relevant_count", "query"], ascending=[False, True]).reset_index(drop=True)
    eval_df.insert(0, "query_id", [f"search_q_{idx + 1:04d}" for idx in range(len(eval_df))])
    if "source_split" not in eval_df.columns:
        eval_df["source_split"] = "catalog_grouped"
    return eval_df


def _image_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _image_endpoint(endpoint: str, product_id: str) -> str:
    parts = urlsplit(endpoint)
    base_path = parts.path or "/search"
    if base_path.endswith("/search"):
        image_path = f"{base_path[:-7]}/api/images/{product_id}"
    else:
        image_path = f"/api/images/{product_id}"
    return urlunsplit((parts.scheme, parts.netloc, image_path, parts.query, parts.fragment))


def _fetch_image_base64(endpoint: str, product_id: str, timeout: float) -> str:
    image_url = _image_endpoint(endpoint, product_id)
    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRIES):
        try:
            with urllib.request.urlopen(image_url, timeout=timeout) as response:
                return base64.b64encode(response.read()).decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {image_url}: {detail}") from exc
        except (urllib.error.URLError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(REQUEST_RETRY_SLEEP_SEC)
                continue
            raise RuntimeError(f"Failed to fetch query image from {image_url}: {exc}") from exc
    raise RuntimeError(f"Failed to fetch query image from {image_url}: {last_error}")


def _image_to_base64_from_pil(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _fallback_image_base64_from_local_engine(product_id: str, mode: str) -> str | None:
    try:
        engine = _get_local_engine(mode)
    except Exception:
        return None

    item = engine.find_item(product_id)
    if item is None:
        return None

    if item.image_path:
        image_file = Path(item.image_path)
        if image_file.exists():
            return _image_to_base64(str(image_file))

    if item.image is not None:
        return _image_to_base64_from_pil(item.image)

    return None


def _post_search(
    endpoint: str,
    query: str,
    image_base64: str | None,
    top_k: int,
    timeout: float,
    use_cache: bool = False,
) -> dict[str, Any]:
    payload = json.dumps(
        {"query": query, "image_base64": image_base64, "top_k": top_k, "use_cache": use_cache}
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
        except (urllib.error.URLError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(REQUEST_RETRY_SLEEP_SEC)
                continue
            raise RuntimeError(f"Failed to connect to {endpoint}: {exc}") from exc
    raise RuntimeError(f"Failed to connect to {endpoint}: {last_error}")


def _health_endpoint(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    path = parts.path or "/search"
    if path.endswith("/search"):
        path = f"{path[:-7]}/health"
    else:
        path = "/health"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _fetch_endpoint_health(endpoint: str, timeout: float) -> dict[str, Any]:
    health_url = _health_endpoint(endpoint)
    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRIES):
        try:
            with urllib.request.urlopen(health_url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {health_url}: {detail}") from exc
        except (urllib.error.URLError, http.client.RemoteDisconnected, ConnectionResetError) as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES - 1:
                time.sleep(REQUEST_RETRY_SLEEP_SEC)
                continue
            raise RuntimeError(
                f"Failed to connect to {health_url}: {exc}. "
                "On this machine, try 127.0.0.1 instead of localhost if the server is already running."
            ) from exc
    raise RuntimeError(f"Failed to connect to {health_url}: {last_error}")


def _infer_mode_from_health(health: dict[str, Any]) -> str | None:
    endpoint_mode = str(health.get("mode", "")).strip().lower()
    if endpoint_mode in {"test", "dev", "production"}:
        return endpoint_mode

    index_size = int(health.get("index_size", 0) or 0)
    if index_size <= 0:
        return None

    candidates: list[str] = []
    for mode_name in ("test", "dev", "production"):
        meta_path = _metadata_path_for_mode(mode_name)
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        item_count = len(payload.get("items", []))
        if item_count == index_size:
            candidates.append(mode_name)

    if len(candidates) == 1:
        return candidates[0]

    if index_size == 500:
        return "test"

    return None


def _resolve_eval_mode(requested_mode: str, endpoint: str, timeout: float) -> tuple[str, dict[str, Any]]:
    health = _fetch_endpoint_health(endpoint, timeout)
    endpoint_mode = _infer_mode_from_health(health)
    if endpoint_mode not in {"test", "dev", "production"}:
        raise RuntimeError(
            "Search endpoint health did not expose a recognizable mode and it could not be inferred. "
            f"Health payload={health!r}. "
            "Rebuild the search-engine container to get the latest /health response, or pass --mode explicitly."
        )

    if requested_mode == "auto":
        return endpoint_mode, health

    if requested_mode != endpoint_mode:
        raise RuntimeError(
            "Evaluation mode does not match the running search engine. "
            f"Requested mode={requested_mode}, endpoint mode={endpoint_mode}, endpoint={endpoint}. "
            "Use --mode to match the server or restart the search engine with the intended mode."
        )

    return requested_mode, health


def _resolve_query_image_base64(row: Any, endpoint: str, timeout: float, mode: str) -> str:
    image_path = str(getattr(row, "query_image_path", "") or "").strip()
    if image_path:
        image_file = Path(image_path)
        if image_file.exists():
            return _image_to_base64(str(image_file))
    product_id = str(getattr(row, "query_product_id", "") or "").strip()
    if not product_id:
        raise RuntimeError("No query image source available for this evaluation row.")
    try:
        return _fetch_image_base64(endpoint, product_id, timeout)
    except RuntimeError:
        fallback = _fallback_image_base64_from_local_engine(product_id, mode)
        if fallback:
            return fallback
        raise


def _mode_payload(modality: str, row: Any, endpoint: str, timeout: float, mode: str) -> tuple[str, str | None]:
    if modality == "text":
        return str(row.query), None
    image_b64 = _resolve_query_image_base64(row, endpoint, timeout, mode)
    if modality == "image":
        return "", image_b64
    return str(row.query), image_b64


def _warm_up_modality(
    eval_df: pd.DataFrame,
    endpoint: str,
    modality: str,
    top_k: int,
    timeout: float,
    mode: str,
) -> None:
    if eval_df.empty:
        return
    first_row = next(eval_df.itertuples(index=False))
    query_text, image_b64 = _mode_payload(modality, first_row, endpoint, timeout, mode)
    _post_search(
        endpoint=endpoint,
        query=query_text,
        image_base64=image_b64,
        top_k=top_k,
        timeout=timeout,
        use_cache=False,
    )


def evaluate_modality(
    eval_df: pd.DataFrame,
    endpoint: str,
    modality: str,
    top_k: int,
    metric_k: int,
    timeout: float,
    mode: str,
) -> dict[str, Any]:
    # Warm up CLIP model/image cache so the report reflects steady-state search latency
    # rather than one-time startup costs from the first multimodal request.
    _warm_up_modality(eval_df, endpoint, modality, top_k, timeout, mode)

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []
    api_latencies: list[float] = []
    wall_latencies: list[float] = []
    rows: list[dict[str, Any]] = []

    for row in eval_df.itertuples(index=False):
        query_text, image_b64 = _mode_payload(modality, row, endpoint, timeout, mode)
        started = time.perf_counter()
        response = _post_search(
            endpoint=endpoint,
            query=query_text,
            image_base64=image_b64,
            top_k=top_k,
            timeout=timeout,
            use_cache=False,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0

        ranked_items = [
            _normalize_article_id(item.get("product_id", ""))
            for item in response.get("results", [])
            if _normalize_article_id(item.get("product_id", ""))
        ]
        relevant_items = [
            _normalize_article_id(item)
            for item in row.relevant_items
            if _normalize_article_id(item)
        ]
        query_product_id = _normalize_article_id(row.query_product_id)

        # For image and hybrid evaluation, the query image usually comes from the same
        # catalog as the indexed items. If we score the identical source product as
        # relevant, the task collapses into exact-image replay and overstates quality.
        # We therefore remove the query product itself and measure whether similar
        # items from the same query group are retrieved.
        if modality in {"image", "hybrid"} and query_product_id:
            filtered_relevant = [item for item in relevant_items if item != query_product_id]
            filtered_ranked = [item for item in ranked_items if item != query_product_id]
            if filtered_relevant:
                relevant_items = filtered_relevant
                ranked_items = filtered_ranked

        ranked_lists.append(ranked_items)
        relevant_lists.append(relevant_items)
        api_latencies.append(float(response.get("latency_ms", wall_ms)))
        wall_latencies.append(wall_ms)
        rows.append(
            {
                "query_id": row.query_id,
                "query": row.query,
                "query_product_id": query_product_id,
                "search_type": response.get("search_type", modality),
                "relevant_items": relevant_items,
                "ranked_items": ranked_items,
                "latency_ms": float(response.get("latency_ms", wall_ms)),
                "wall_latency_ms": wall_ms,
            }
        )

    hitrate = mean_hit_rate_at_k(ranked_lists, relevant_lists, metric_k)
    mrr = mean_reciprocal_rank(ranked_lists, relevant_lists)
    ndcg = mean_ndcg_at_k(ranked_lists, relevant_lists, metric_k)
    avg_api_latency = mean(api_latencies)
    avg_wall_latency = mean(wall_latencies)
    p95_api_latency = float(pd.Series(api_latencies).quantile(0.95))
    p95_wall_latency = float(pd.Series(wall_latencies).quantile(0.95))

    return {
        "samples_evaluated": len(rows),
        f"HitRate@{metric_k}": hitrate,
        "MRR": mrr,
        f"NDCG@{metric_k}": ndcg,
        "avg_latency_ms": avg_api_latency,
        "avg_api_latency_ms": avg_api_latency,
        "p95_latency_ms": p95_api_latency,
        "avg_wall_latency_ms": avg_wall_latency,
        "p95_wall_latency_ms": p95_wall_latency,
        "checks": {
            "latency_within_200ms": avg_api_latency <= 200.0,
            "mrr_meets_target": mrr >= 0.55,
            "ndcg_meets_target": ndcg >= 0.50,
        },
        "rows": rows,
    }


class SimpleBM25:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.tokenized_docs = [_tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freqs: Counter[str] = Counter()
        for tokens in self.tokenized_docs:
            for token in set(tokens):
                self.doc_freqs[token] += 1
        self.k1 = 1.5
        self.b = 0.75

    def scores(self, query: str) -> list[float]:
        tokens = _tokenize(query)
        scores = [0.0 for _ in self.documents]
        if not tokens:
            return scores

        num_docs = len(self.documents)
        for token in tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
            for index, doc_tokens in enumerate(self.tokenized_docs):
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lengths[index] / max(self.avgdl, 1e-9)))
                scores[index] += idf * (tf * (self.k1 + 1) / denom)
        return scores


def evaluate_bm25_baseline(
    eval_df: pd.DataFrame,
    items_df: pd.DataFrame,
    metric_k: int,
) -> dict[str, float]:
    corpus_df = items_df.copy()
    corpus_df["document"] = corpus_df.apply(
        lambda row: " ".join(
            filter(
                None,
                [
                    str(row.get("name", "")).strip(),
                    str(row.get("description", "")).strip(),
                    str(row.get("product_type_name", "")).strip(),
                    str(row.get("colour_group_name", "")).strip(),
                ],
            )
        ).strip(),
        axis=1,
    )
    corpus_df = corpus_df[corpus_df["product_id"].astype(str).str.strip() != ""].reset_index(drop=True)
    bm25 = SimpleBM25(corpus_df["document"].astype(str).tolist())

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []

    for row in eval_df.itertuples(index=False):
        scores = bm25.scores(str(row.query))
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:metric_k]
        ranked_items = [
            _normalize_article_id(corpus_df.iloc[idx]["product_id"])
            for idx in ranked_indices
            if _normalize_article_id(corpus_df.iloc[idx]["product_id"])
        ]
        ranked_lists.append(ranked_items)
        relevant_lists.append(
            [_normalize_article_id(item) for item in row.relevant_items if _normalize_article_id(item)]
        )

    return {
        "MRR": mean_reciprocal_rank(ranked_lists, relevant_lists),
        f"NDCG@{metric_k}": mean_ndcg_at_k(ranked_lists, relevant_lists, metric_k),
        f"HitRate@{metric_k}": mean_hit_rate_at_k(ranked_lists, relevant_lists, metric_k),
    }


def _overall_summary(modalities: dict[str, dict[str, Any]], metric_k: int) -> tuple[dict[str, Any], dict[str, bool]]:
    total_samples = sum(int(result.get("samples_evaluated", 0) or 0) for result in modalities.values())
    if total_samples <= 0:
        total_samples = max(len(modalities), 1)

    def _weighted_average(field: str) -> float:
        weighted_sum = 0.0
        for result in modalities.values():
            weight = int(result.get("samples_evaluated", 0) or 0) or 1
            weighted_sum += float(result[field]) * weight
        return weighted_sum / total_samples

    avg_mrr = _weighted_average("MRR")
    avg_ndcg = _weighted_average(f"NDCG@{metric_k}")
    avg_latency = _weighted_average("avg_latency_ms")
    avg_hit_rate = _weighted_average(f"HitRate@{metric_k}")
    p95_latency = max(result["p95_latency_ms"] for result in modalities.values())
    summary = {
        f"HitRate@{metric_k}": avg_hit_rate,
        "MRR": avg_mrr,
        f"NDCG@{metric_k}": avg_ndcg,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
    }
    checks = {
        "latency_within_200ms": avg_latency <= 200.0,
        "mrr_meets_target": avg_mrr >= 0.55,
        "ndcg_meets_target": avg_ndcg >= 0.50,
    }
    return summary, checks


def _load_split_summary(mode: str) -> dict[str, Any]:
    if mode == "dev" and DEV_SPLIT_SUMMARY_PATH.exists():
        try:
            return json.loads(DEV_SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def write_markdown_report(
    report: dict[str, Any],
    output_path: Path,
    eval_df: pd.DataFrame,
) -> None:
    metadata = report["metadata"]
    metric_k = metadata["metric_k"]
    modality_labels = {
        "text": "텍스트",
        "image": "이미지",
        "hybrid": "하이브리드",
    }
    lines: list[str] = []
    lines.append("# 검색 엔진 실험 리포트")
    lines.append("")
    lines.append(f"- 생성 시각: {metadata['generated_at']}")
    lines.append(f"- 실행 모드: `{metadata['mode']}`")
    lines.append(f"- 검색 엔드포인트: `{metadata['endpoint']}`")
    lines.append(f"- 재현용 시드: `{metadata['random_seed']}`")
    lines.append(f"- dev 샘플 크기: `{metadata['sample_size']}`")
    lines.append(f"- 평가된 쿼리 그룹 수: `{metadata['samples_evaluated']}`")
    lines.append("")
    lines.append("## 데이터 구성 및 분할")
    lines.append("")
    lines.append("- 원본 상품 카탈로그: `data/raw/articles.csv`")
    lines.append("- 이미지 소스: 설정된 이미지 루트의 H&M 상품 이미지 파일")
    lines.append("- 검색 검증용 코퍼스: `random_seed=42`로 고정 추출한 `dev` 이미지 포함 subset")
    lines.append("- 쿼리 구성 방식: `colour_group_name + product_type_name` 기반 canonical query로 그룹화하고, 관련 상품이 2개 이상이며 쿼리용 이미지가 1개 이상 있는 그룹만 사용")
    lines.append("- 추천/시뮬레이션용 이벤트 분할은 프로젝트 공통 규격인 시간 기반 `train / valid / test = 8 / 1 / 1`을 그대로 사용")
    lines.append("")
    lines.append("## 지표 정의")
    lines.append("")
    lines.append(f"- Offline top-k 평가 기준: `k={metric_k}`")
    lines.append("- `MRR`: 첫 번째 관련 상품이 나타난 순위의 역수 평균")
    lines.append(f"- `NDCG@{metric_k}`: 상위 {metric_k}개 결과에 대한 정규화 누적 이득")
    lines.append("- `HitRate@k`: 상위 k개 안에 관련 상품이 하나라도 포함된 비율")
    lines.append("- `Latency`: 서버 warmup 이후 요청 단위 wall-clock 응답 시간")
    lines.append("")
    lines.append("## 모달리티별 결과")
    lines.append("")
    lines.append("| 모달리티 | 샘플 수 | MRR | NDCG@10 | HitRate@10 | 평균 지연시간 (ms) | P95 지연시간 (ms) | 기준 통과 여부 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for modality, result in report["modalities"].items():
        checks = result["checks"]
        status = "PASS" if all(checks.values()) else "FAIL"
        lines.append(
            f"| {modality_labels.get(modality, modality)} | {result['samples_evaluated']} | {result['MRR']:.4f} | {result[f'NDCG@{metric_k}']:.4f} | "
            f"{result[f'HitRate@{metric_k}']:.4f} | {result['avg_wall_latency_ms']:.2f} | {result['p95_wall_latency_ms']:.2f} | {status} |"
        )
    lines.append("")
    lines.append("## 베이스라인 비교")
    lines.append("")
    lines.append("- 베이스라인: 동일한 `dev` subset에 대한 BM25 텍스트 단독 검색")
    lines.append(f"- BM25 MRR: `{report['baseline']['text']['MRR']:.4f}`")
    lines.append(f"- BM25 NDCG@{metric_k}: `{report['baseline']['text'][f'NDCG@{metric_k}']:.4f}`")
    lines.append(f"- CLIP 텍스트 검색의 MRR 개선폭: `{report['baseline']['text_improvement']['MRR_delta']:+.4f}`")
    lines.append(f"- CLIP 텍스트 검색의 NDCG@{metric_k} 개선폭: `{report['baseline']['text_improvement'][f'NDCG@{metric_k}_delta']:+.4f}`")
    lines.append("")
    lines.append("## 재현성 설정")
    lines.append("")
    lines.append("- CLIP 체크포인트: `openai/clip-vit-base-patch32`")
    lines.append("- FAISS 인덱스: L2 normalize 후 inner product를 사용하는 `IndexHNSWFlat`")
    lines.append("- 쿼리 셋 생성 시드: `42`")
    lines.append("- dev subset은 이미지가 실제로 존재하는 상품만 사용하여 텍스트, 이미지, 하이브리드 검색을 같은 searchable universe에서 비교")
    lines.append("")
    lines.append("## 해석 메모")
    lines.append("")
    lines.append("- 본 리포트는 전체 이미지 데이터셋으로 production 인덱스를 만들지 않고도 멀티모달 검색 동작을 검증하기 위한 `dev` 모드 실험 결과입니다.")
    lines.append("- 전체 데이터셋 기준의 최종 평가는 더 긴 인덱싱 시간을 허용할 수 있을 때 별도로 수행하는 것이 적절합니다.")
    lines.append("")
    lines.append("## 쿼리 미리보기")
    lines.append("")
    preview = eval_df.head(10)[["query_id", "query", "query_product_id", "relevant_count"]]
    lines.append("| query_id | query | 대표 상품 id | 관련 상품 수 |")
    lines.append("| --- | --- | --- | ---: |")
    for row in preview.itertuples(index=False):
        lines.append(
            f"| {row.query_id} | {str(row.query).replace('|', '/')} | {row.query_product_id} | {row.relevant_count} |"
        )
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_search_markdown_report_v2(
    report: dict[str, Any],
    output_path: Path,
    eval_df: pd.DataFrame,
) -> None:
    metadata = report["metadata"]
    metric_k = metadata["metric_k"]
    split_summary = report.get("split_summary", {})
    modality_labels = {
        "text": "텍스트",
        "image": "이미지",
        "hybrid": "하이브리드",
    }

    lines: list[str] = []
    lines.append("# 검색 엔진 실험 리포트")
    lines.append("")
    lines.append("## 실행 정보")
    lines.append("")
    lines.append(f"- 생성 시각: {metadata['generated_at']}")
    lines.append(f"- 실행 모드: `{metadata['mode']}`")
    lines.append(f"- 검색 엔드포인트: `{metadata['endpoint']}`")
    lines.append(f"- 재현 시드: `{metadata['random_seed']}`")
    lines.append(f"- 평가 샘플 수: `{metadata['samples_evaluated']}`")
    lines.append(f"- 원본 샘플링 상한: `{metadata['sample_size']}`")
    lines.append("")
    lines.append("## 데이터 구성 및 분할")
    lines.append("")
    lines.append("- 상품 메타데이터: `data/raw/articles.csv`")
    lines.append("- 이미지 데이터: `D:/imagedata/<첫 세 자리>/<article_id>.jpg`")
    lines.append("- 검색 인덱스 입력: raw article 메타데이터와 이미지 경로를 매칭한 멀티모달 catalog")
    lines.append("- 분할 방식: 시간 기반 `train / valid / test = 8 / 1 / 1`")
    if split_summary:
        ratios = split_summary.get("ratios", {})
        lines.append(
            f"- split summary: train={ratios.get('train', 'n/a')}, valid={ratios.get('valid', 'n/a')}, test={ratios.get('test', 'n/a')}"
        )
        test_info = split_summary.get("test", {})
        if test_info:
            lines.append(
                f"- test split rows={test_info.get('row_count', 'n/a')}, unique_sessions={test_info.get('unique_sessions', 'n/a')}, "
                f"range={test_info.get('first_timestamp', 'n/a')} ~ {test_info.get('last_timestamp', 'n/a')}"
            )
    lines.append("- 평가셋 구성: 현재 인덱스에 실제로 존재하는 상품을 `color + product type` 기준으로 그룹화해 query bucket을 생성")
    lines.append("- relevance 정의: 같은 query bucket에 속한 모든 indexed 상품을 relevant item으로 간주")
    lines.append("- 이미지/하이브리드 평가는 query로 사용한 동일 상품을 가능한 경우 점수 계산에서 제외해, exact-image replay 대신 similar-item retrieval을 측정")
    lines.append("")
    lines.append("## 평가 기준")
    lines.append("")
    lines.append(f"- Offline top-k 기준: `k={metric_k}`")
    lines.append("- 응답 시간 목표: 평균 API latency `<= 200ms`")
    lines.append("- 목표 정확도: `MRR >= 0.55`, `NDCG@10 >= 0.50`")
    lines.append("- 검색 평가는 retrieval top-k 기반이므로 별도 negative sampling 없이 전체 searchable universe 대비 측정")
    lines.append("")
    lines.append("## 지표 정의")
    lines.append("")
    lines.append("- `MRR`: 첫 relevant item의 역순위 평균")
    lines.append(f"- `NDCG@{metric_k}`: 상위 {metric_k}개 결과의 정규화 누적 이득")
    lines.append(f"- `HitRate@{metric_k}`: 상위 {metric_k}개 결과 안에 relevant item이 하나라도 포함될 비율")
    lines.append("- `API Latency`: `/search` 응답 본문에 기록된 서버 측 처리 시간")
    lines.append("- `Wall Latency`: 평가 스크립트에서 관측한 클라이언트 왕복 시간")
    lines.append("")
    lines.append("## 모달리티별 결과")
    lines.append("")
    lines.append("| 모달리티 | 샘플 수 | MRR | NDCG@10 | HitRate@10 | 평균 API Latency(ms) | P95 API Latency(ms) | 평균 Wall Latency(ms) | 목표 통과 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for modality, result in report["modalities"].items():
        status = "PASS" if all(result["checks"].values()) else "FAIL"
        lines.append(
            f"| {modality_labels.get(modality, modality)} | {result['samples_evaluated']} | {result['MRR']:.4f} | "
            f"{result[f'NDCG@{metric_k}']:.4f} | {result[f'HitRate@{metric_k}']:.4f} | "
            f"{result['avg_latency_ms']:.2f} | {result['p95_latency_ms']:.2f} | {result['avg_wall_latency_ms']:.2f} | {status} |"
        )
    lines.append("")
    lines.append("## 베이스라인 비교")
    lines.append("")
    lines.append("- 베이스라인: BM25 text-only")
    lines.append(f"- BM25 MRR: `{report['baseline']['text']['MRR']:.4f}`")
    lines.append(f"- BM25 NDCG@{metric_k}: `{report['baseline']['text'][f'NDCG@{metric_k}']:.4f}`")
    lines.append(f"- CLIP text MRR 개선폭: `{report['baseline']['text_improvement']['MRR_delta']:+.4f}`")
    lines.append(f"- CLIP text NDCG@{metric_k} 개선폭: `{report['baseline']['text_improvement'][f'NDCG@{metric_k}_delta']:+.4f}`")
    lines.append("")
    lines.append("## 재현 설정")
    lines.append("")
    lines.append("- CLIP checkpoint: `openai/clip-vit-base-patch32`")
    lines.append("- Index: `FAISS IndexHNSWFlat` + cosine-style inner product")
    lines.append("- Random seed: `42`")
    lines.append("- Dev index는 실제 이미지가 존재하는 상품만 사용")
    lines.append("")
    lines.append("## 쿼리 미리보기")
    lines.append("")
    preview = eval_df.head(10)[["query_id", "query", "query_product_id", "relevant_count", "source_split"]]
    lines.append("| query_id | query | 대표 상품 id | relevant 수 | source |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for row in preview.itertuples(index=False):
        lines.append(
            f"| {row.query_id} | {str(row.query).replace('|', '/')} | {row.query_product_id} | {row.relevant_count} | {row.source_split} |"
        )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multimodal search quality report for the dashboard")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--mode", choices=["auto", "test", "dev", "production"], default=DEFAULT_MODE)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--min-relevant", type=int, default=DEFAULT_MIN_RELEVANT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--metric-k", type=int, default=DEFAULT_METRIC_K)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--eval-set-output", default=str(SEARCH_EVAL_SET_PATH))
    parser.add_argument("--report-output", default=str(SEARCH_JSON_REPORT_PATH))
    parser.add_argument("--doc-output", default=str(SEARCH_DOC_REPORT_PATH))
    args = parser.parse_args()

    resolved_mode, endpoint_health = _resolve_eval_mode(
        requested_mode=args.mode,
        endpoint=args.endpoint,
        timeout=args.timeout,
    )

    eval_df = build_eval_set(
        mode=resolved_mode,
        sample_size=args.sample_size,
        min_relevant=args.min_relevant,
        random_seed=args.random_seed,
    )
    eval_output = Path(args.eval_set_output)
    report_output = Path(args.report_output)
    doc_output = Path(args.doc_output)

    eval_output.parent.mkdir(parents=True, exist_ok=True)
    eval_df.assign(relevant_items=eval_df["relevant_items"].apply(lambda items: "|".join(items))).to_csv(
        eval_output,
        index=False,
        encoding="utf-8-sig",
    )

    modalities = {
        modality: evaluate_modality(
            eval_df=eval_df,
            endpoint=args.endpoint,
            modality=modality,
            top_k=args.top_k,
            metric_k=args.metric_k,
            timeout=args.timeout,
            mode=resolved_mode,
        )
        for modality in ("text", "image", "hybrid")
    }
    overall_search, overall_checks = _overall_summary(modalities, args.metric_k)
    items_df = _load_index_items(resolved_mode)
    baseline_text = evaluate_bm25_baseline(eval_df=eval_df, items_df=items_df, metric_k=args.metric_k)

    report = {
        "metadata": {
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "endpoint": args.endpoint,
            "mode": resolved_mode,
            "top_k": args.top_k,
            "metric_k": args.metric_k,
            "samples_evaluated": len(eval_df),
            "sample_size": args.sample_size,
            "random_seed": args.random_seed,
            "endpoint_health": endpoint_health,
            "eval_source_counts": eval_df["source_split"].value_counts().to_dict() if "source_split" in eval_df.columns else {},
            "query_cache_mode": "disabled_during_evaluation",
        },
        "split_summary": _load_split_summary(resolved_mode),
        "thresholds": {
            "latency_ms_max": 200.0,
            "mrr_min": 0.55,
            "ndcg_at_10_min": 0.50,
        },
        "search": overall_search,
        "checks": overall_checks,
        "modalities": modalities,
        "baseline": {
            "name": "BM25 text-only",
            "text": baseline_text,
            "text_improvement": {
                "MRR_delta": modalities["text"]["MRR"] - baseline_text["MRR"],
                f"NDCG@{args.metric_k}_delta": modalities["text"][f"NDCG@{args.metric_k}"] - baseline_text[f"NDCG@{args.metric_k}"],
            },
        },
    }

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_search_markdown_report_v2(report=report, output_path=doc_output, eval_df=eval_df)

    print("===== Search Metrics Report =====")
    print(f"Eval set           : {eval_output}")
    print(f"JSON report        : {report_output}")
    print(f"Doc report         : {doc_output}")
    print(f"Samples evaluated  : {len(eval_df)}")
    print(
        f"[overall] MRR={overall_search['MRR']:.4f} "
        f"NDCG@{args.metric_k}={overall_search[f'NDCG@{args.metric_k}']:.4f} "
        f"AvgLatency={overall_search['avg_latency_ms']:.2f}ms"
    )
    for modality, result in modalities.items():
        print(
            f"[{modality}] MRR={result['MRR']:.4f} "
            f"NDCG@{args.metric_k}={result[f'NDCG@{args.metric_k}']:.4f} "
            f"AvgLatency={result['avg_latency_ms']:.2f}ms"
        )
    print(
        f"[baseline:text] MRR={baseline_text['MRR']:.4f} "
        f"NDCG@{args.metric_k}={baseline_text[f'NDCG@{args.metric_k}']:.4f}"
    )
    print(f"All latency <= 200ms : {'PASS' if overall_checks['latency_within_200ms'] else 'FAIL'}")
    print(f"All MRR >= 0.55      : {'PASS' if overall_checks['mrr_meets_target'] else 'FAIL'}")
    print(f"All NDCG >= 0.50     : {'PASS' if overall_checks['ndcg_meets_target'] else 'FAIL'}")


if __name__ == "__main__":
    main()
