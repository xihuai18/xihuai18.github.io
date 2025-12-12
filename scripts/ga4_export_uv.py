#!/usr/bin/env python3

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_post_dates(posts_dir: Path) -> Iterable[date]:
    if not posts_dir.exists():
        return

    for path in posts_dir.glob("*.md"):
        # Jekyll post filenames typically start with YYYY-MM-DD-
        name = path.name
        if len(name) < 10:
            continue
        prefix = name[:10]
        try:
            yield datetime.strptime(prefix, "%Y-%m-%d").date()
        except ValueError:
            continue


def _default_start_date() -> str:
    root = _repo_root()
    dates = sorted(_iter_post_dates(root / "_posts"))
    if dates:
        return dates[0].isoformat()
    return "2020-01-01"


def _normalize_page_path(path: str) -> str:
    # Keep GA4 pagePath style: leading '/', no scheme/host.
    if not path:
        return ""
    if not path.startswith("/"):
        path = "/" + path
    # Normalize common variants.
    if path.endswith("/index.html"):
        path = path[: -len("index.html")]
    if path.endswith("index.html") and path != "/index.html":
        path = path[: -len("index.html")]
    return path


def fetch_page_uv(property_id: str, start_date: str, end_date: str = "today") -> Dict[str, int]:
    # Lazy import so the script can print a clearer message if deps are missing.
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
        from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: google-analytics-data. " "Install with: pip install google-analytics-data"
        ) from exc

    client = BetaAnalyticsDataClient()

    page_uv: Dict[str, int] = {}

    limit = 10000
    offset = 0

    while True:
        request = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=[Dimension(name="pagePath")],
            metrics=[Metric(name="totalUsers")],
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            limit=limit,
            offset=offset,
        )

        response = client.run_report(request)

        if not response.rows:
            break

        for row in response.rows:
            page_path = _normalize_page_path(row.dimension_values[0].value)
            if not page_path:
                continue

            # totalUsers is returned as a string.
            try:
                uv = int(float(row.metric_values[0].value))
            except ValueError:
                continue

            # If the same key appears, keep the max.
            prev = page_uv.get(page_path)
            page_uv[page_path] = uv if prev is None else max(prev, uv)

        offset += limit

    return page_uv


def write_outputs(page_uv_all: Dict[str, int], start_date_all: str, page_uv_30d: Dict[str, int]) -> Tuple[Path, Path]:
    root = _repo_root()
    data_dir = root / "_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    uv_path = data_dir / "post_uv.json"
    meta_path = data_dir / "post_uv_meta.json"

    ordered_all = dict(sorted(page_uv_all.items(), key=lambda kv: kv[0]))
    ordered_30d = dict(sorted(page_uv_30d.items(), key=lambda kv: kv[0]))

    uv_path.write_text(
        json.dumps({"all": ordered_all, "d30": ordered_30d}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    meta = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "metric": "totalUsers",
        "dimension": "pagePath",
        "all": {"start_date": start_date_all, "end_date": "today"},
        "d30": {"start_date": "30daysAgo", "end_date": "today"},
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return uv_path, meta_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-page UV from GA4 to Jekyll _data JSON")
    parser.add_argument("--property-id", required=False, help="GA4 Property ID (numeric)")
    parser.add_argument(
        "--start-date",
        required=False,
        help="Start date (YYYY-MM-DD). Default: earliest _posts date, else 2020-01-01.",
    )

    args = parser.parse_args()

    property_id = args.property_id or os.getenv("GA4_PROPERTY_ID")
    if not property_id:
        raise SystemExit("Missing GA4 property id. Provide --property-id or set GA4_PROPERTY_ID.")

    start_date = args.start_date or os.getenv("GA4_START_DATE") or _default_start_date()

    page_uv_all = fetch_page_uv(property_id=property_id, start_date=start_date)
    page_uv_30d = fetch_page_uv(property_id=property_id, start_date="30daysAgo")
    uv_path, meta_path = write_outputs(page_uv_all, start_date_all=start_date, page_uv_30d=page_uv_30d)

    print(f"Wrote all={len(page_uv_all)} and d30={len(page_uv_30d)} rows to {uv_path}")
    print(f"Wrote meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
