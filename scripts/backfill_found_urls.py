"""Заполняет found_urls в data/results/<config>.csv из <config>_ctx.jsonl.

Нужен для прогонов B, C и D от 2026-09-29: фронтенд Langflow при сохранении потока удалил связь
QdrantSearch.sources -> FormatAnswer.sources, поэтому поток вернул ответы без списка найденных страниц.
На сами ответы это не повлияло: контекст передавался в промпт по другой связи. В *_ctx.jsonl лежат
топ-5 чанков прямого запроса к Qdrant с тем же префиксом, что в потоке; на шаге 4 проверено, что
поток находит те же страницы в том же порядке.

    python scripts/backfill_found_urls.py B C D
"""
import csv
import json
import sys

from common import ROOT

RESULTS = ROOT / "data" / "results"

for config in sys.argv[1:]:
    path = RESULTS / f"{config}.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ctx = {c["id"]: c for c in map(json.loads, (RESULTS / f"{config}_ctx.jsonl").open(encoding="utf-8"))}
    filled = 0
    for r in rows:
        if r["mode"] == "rag" and not r["found_urls"]:
            urls = []
            for c in ctx[r["id"]]["chunks"]:
                if c["url"] not in urls:
                    urls.append(c["url"])
            r["found_urls"] = " ".join(urls)
            filled += 1
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{config}: заполнено {filled} из {len(rows)}")
