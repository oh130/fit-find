from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def build_payload(image_path: Path, query: str, top_k: int) -> dict[str, object]:
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "query": query,
        "image_base64": image_base64,
        "top_k": int(top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode an image as base64 and build a JSON payload for POST /api/search"
    )
    parser.add_argument("--image", required=True, help="Path to the query image file")
    parser.add_argument("--query", default="", help="Optional text query for hybrid search")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k results to request")
    parser.add_argument("--output", help="Optional file path to save the JSON payload")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    payload = build_payload(image_path=image_path, query=args.query, top_k=args.top_k)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload_json, encoding="utf-8")
        print(f"Saved payload to: {output_path}")
    else:
        print(payload_json)


if __name__ == "__main__":
    main()
