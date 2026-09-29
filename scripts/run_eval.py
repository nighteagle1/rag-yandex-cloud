"""Прогоняет тестовые вопросы через поток Langflow и сохраняет ответы в data/results/<config>.csv.

Конфигурации (все с температурой 0):
    A — без поиска (тот же поток, шаблон prompts/norag.md без контекста)
    B — yc_c1000, промпт v1
    C — yc_c1000, промпт v2
    D — yc_c500, промпт v2

    python scripts/run_eval.py --config C --dry-run
    python scripts/run_eval.py --config C
    python scripts/run_eval.py --config C --run 2      # повтор -> data/results/C_run2.csv

Для rag отдельно считаются hit@5 и reciprocal rank по совпадению URL найденных чанков с gold_urls
(запрос в Qdrant напрямую, тот же префикс запроса, что и в потоке). Уже посчитанные строки не
перезапрашиваются, поэтому прерванный прогон можно продолжить той же командой.
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from tqdm import tqdm

from common import ANSWER_MODEL, ROOT, load_prompt, model_price, run_flow, search

CONFIGS = {
    "A": {"mode": "norag", "collection": None, "prompt": "norag"},
    "B": {"mode": "rag", "collection": "yc_c1000", "prompt": "v1"},
    "C": {"mode": "rag", "collection": "yc_c1000", "prompt": "v2"},
    "D": {"mode": "rag", "collection": "yc_c500", "prompt": "v2"},
}
QUESTIONS = ROOT / "data" / "questions.csv"
RESULTS = ROOT / "data" / "results"
FIELDS = ["id", "split", "type", "should_refuse", "question", "answer", "found_urls", "hit5", "rr",
          "latency_s", "trace_id", "config", "mode", "collection", "prompt"]
ENC = tiktoken.get_encoding("cl100k_base")


def load_questions(split: str) -> list[dict]:
    rows = list(csv.DictReader(QUESTIONS.open(encoding="utf-8")))
    return [r for r in rows if split == "all" or r["split"] == split]


def retrieval_metrics(question: str, gold_urls: list[str], collection: str) -> tuple[list[dict], int | str, float | str]:
    """hit@5 и reciprocal rank по чанкам топ-5; для вопросов без gold_urls метрики не считаются."""
    hits = search(question, collection, top_k=5)
    if not gold_urls:
        return hits, "", ""
    rank = next((i for i, h in enumerate(hits, 1) if h["url"] in gold_urls), None)
    return hits, int(rank is not None), round(1 / rank, 4) if rank else 0.0


def run_one(q: dict, name: str, cfg: dict) -> tuple[dict, dict]:
    res = run_flow(q["question"], collection=cfg["collection"], prompt_text=load_prompt(cfg["prompt"]),
                   session_id=f"eval-{name}")
    row = {"id": q["id"], "split": q["split"], "type": q["type"], "should_refuse": q["should_refuse"],
           "question": q["question"], "answer": res["answer"], "latency_s": res["latency_s"],
           "trace_id": res["trace_id"], "config": name, "mode": cfg["mode"],
           "collection": cfg["collection"] or "", "prompt": cfg["prompt"], "found_urls": "", "hit5": "", "rr": ""}
    ctx = {"id": q["id"], "chunks": []}
    if cfg["mode"] == "rag":
        row["found_urls"] = " ".join(res["urls"])
        hits, row["hit5"], row["rr"] = retrieval_metrics(q["question"], q["gold_urls"].split(), cfg["collection"])
        # найденный контекст нужен судье; порядок и состав совпадают с потоком (проверено на шаге 4)
        ctx["chunks"] = [{"url": h["url"], "text": h["text"]} for h in hits]
    return row, ctx


def estimate(questions, cfg):
    p_in, p_out = model_price(ANSWER_MODEL)
    prompt_tokens = len(ENC.encode(load_prompt(cfg["prompt"])))
    ctx_tokens = {"yc_c1000": 5 * 1000, "yc_c500": 5 * 500, None: 0}[cfg["collection"]]
    n_in = sum(prompt_tokens + ctx_tokens + len(ENC.encode(q["question"])) for q in questions)
    n_out = 250 * len(questions)
    # x2: токенизатор qwen считает русский текст дороже cl100k (замер на шаге 3)
    cost = 2 * (n_in * p_in + n_out * p_out) / 1e6
    print(f"{len(questions)} вопросов, {ANSWER_MODEL}: ~{n_in:,} вх. / ~{n_out:,} вых. токенов (cl100k), "
          f"оценка с x2: ${cost:.4f} (+ эмбеддинги запросов, < $0.001)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--run", type=int, default=1, help="номер повтора; 1 — основной прогон")
    ap.add_argument("--split", default="all", choices=["all", "dev", "holdout"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    name = args.config if args.run == 1 else f"{args.config}_run{args.run}"
    out, ctx_out = RESULTS / f"{name}.csv", RESULTS / f"{name}_ctx.jsonl"
    questions = load_questions(args.split)
    done = {r["id"]: r for r in csv.DictReader(out.open(encoding="utf-8"))} if out.exists() else {}
    todo = [q for q in questions if q["id"] not in done]
    print(f"config {name}: {cfg}; готово {len(done)}, осталось {len(todo)}")
    estimate(todo, cfg)
    if args.dry_run or not todo:
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, ctxs = dict(done), {}
    if ctx_out.exists():
        ctxs = {c["id"]: c for c in map(json.loads, ctx_out.open(encoding="utf-8"))}
    errors = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_one, q, name, cfg): q["id"] for q in todo}
        for f in tqdm(as_completed(futures), total=len(futures), desc=name):
            try:
                row, ctx = f.result()
            except Exception as e:  # ошибка одного вопроса не должна ронять прогон; повторный запуск его доберёт
                errors.append((futures[f], str(e)[:200]))
                continue
            rows[row["id"]], ctxs[ctx["id"]] = row, ctx

    order = [q["id"] for q in load_questions("all")]
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for qid in order:
            if qid in rows:
                w.writerow({k: rows[qid].get(k, "") for k in FIELDS})
    with ctx_out.open("w", encoding="utf-8") as fh:
        for qid in order:
            if qid in ctxs:
                fh.write(json.dumps(ctxs[qid], ensure_ascii=False) + "\n")

    scored = [r for r in rows.values() if str(r["hit5"]) != ""]
    if scored:
        hit = sum(int(r["hit5"]) for r in scored) / len(scored)
        mrr = sum(float(r["rr"]) for r in scored) / len(scored)
        print(f"hit@5 = {hit:.3f}, MRR = {mrr:.3f} (вопросов с gold_urls: {len(scored)})")
    print(f"сохранено {len(rows)} строк -> {out}")
    if errors:
        print(f"ошибки ({len(errors)}), перезапустите команду, чтобы добрать:")
        for qid, e in errors:
            print(f"  {qid}: {e}")


if __name__ == "__main__":
    main()
