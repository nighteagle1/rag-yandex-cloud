"""Проверяет, что трейсы прогонов из data/results/*.csv есть в Langfuse.

    python scripts/check_traces.py [--hours 48]

Используется GET /api/public/v2/observations: старый /api/public/traces для организаций,
созданных после 2026-09-16, возвращает 410. Трейсы появляются в Langfuse с задержкой в несколько минут.
"""
import argparse
import csv
import datetime as dt
import os

import requests

from common import ROOT

RESULTS = ROOT / "data" / "results"


def langfuse_trace_ids(hours: float) -> set[str]:
    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    auth = (os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY"))
    now = dt.datetime.now(dt.timezone.utc)
    fmt = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"fromStartTime": fmt(now - dt.timedelta(hours=hours)), "toStartTime": fmt(now),
              "limit": 100, "name": "yc-support-rag"}
    ids, cursor = set(), None
    while True:
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{host}/api/public/v2/observations", params=params, auth=auth, timeout=60)
        r.raise_for_status()
        d = r.json()
        ids |= {o["traceId"] for o in d.get("data", [])}
        cursor = (d.get("meta") or {}).get("cursor")
        if not cursor or not d.get("data"):
            return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=48, help="окно поиска трейсов назад от текущего момента")
    args = ap.parse_args()
    in_langfuse = langfuse_trace_ids(args.hours)
    print(f"трейсов потока в Langfuse за {args.hours:g} ч: {len(in_langfuse)}")
    for path in sorted(RESULTS.glob("*.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows or "trace_id" not in rows[0] or path.stem.endswith("_judged"):
            continue
        ids = [r["trace_id"] for r in rows]
        print(f"  {path.stem}: строк {len(ids)}, уникальных trace_id {len(set(ids))}, "
              f"найдено в Langfuse {len(set(ids) & in_langfuse)}")


if __name__ == "__main__":
    main()
