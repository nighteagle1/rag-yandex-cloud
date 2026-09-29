"""Судья: ставит метки ответам из data/results/<config>.csv и считает метрики.

    python scripts/judge.py --config C --dry-run      # оценка стоимости
    python scripts/judge.py --config C                # метки -> data/results/C_judged.csv
    python scripts/judge.py --report A B C D          # таблица метрик (dev / holdout) и разбивка по типам
    python scripts/judge.py --sample-manual B C D     # 20 случайных строк -> data/results/manual_check.csv
    python scripts/judge.py --agreement               # доля совпадений ручных меток с метками судьи

Метки и рубрика — prompts/judge.md. Метрики (пороги в docs/requirements.md):
    correct      — доля `верно` среди вопросов с ответом в базе (should_refuse=false)
    refusal_ok   — доля `корректный_отказ` среди вопросов без ответа и не по теме
    hallucinated — доля `галлюцинация` среди всех вопросов
    link         — доля ответов со ссылкой yandex.cloud/ru/docs среди вопросов с ответом
    link_found   — доля ответов, где все ссылки взяты из найденных страниц (среди ответов со ссылками, rag)
    hit5, mrr    — поиск, среди вопросов с gold_urls
"""
import argparse
import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from tqdm import tqdm

from common import ROOT, STRONG_MODEL, chat, load_prompt, model_price

RESULTS = ROOT / "data" / "results"
QUESTIONS = ROOT / "data" / "questions.csv"
MANUAL = RESULTS / "manual_check.csv"
LABELS = ["верно", "частично", "неверно", "галлюцинация", "корректный_отказ", "некорректный_отказ"]
REFUSE_LABELS = {"корректный_отказ", "неверно", "галлюцинация"}
ANSWER_LABELS = set(LABELS) - {"корректный_отказ"}
URL_RE = re.compile(r"https://yandex\.cloud/ru/docs/[^\s)\]»\"'<>,;]+")
ENC = tiktoken.get_encoding("cl100k_base")
MANUAL_N, SEED = 20, 7


def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def answer_links(answer: str) -> list[str]:
    """Ссылки на документацию в ответе, без якоря: найденные URL хранятся без него."""
    return [u.rstrip(".").split("#")[0] for u in URL_RE.findall(answer)]


def judge_prompt(row, gold, ctx) -> str:
    chunks = ctx.get("chunks") if ctx else None
    context = ("\n\n---\n\n".join(f"[{c['url']}]\n{c['text']}" for c in chunks) if chunks
               else "(контекст не использовался: режим без поиска)")
    return load_prompt("judge").format(
        type=row["type"], should_refuse=row["should_refuse"], question=row["question"],
        gold_answer=gold["gold_answer"], gold_urls=gold["gold_urls"] or "—", context=context,
        answer=row["answer"] or "(пустой ответ)")


def judge_one(row, gold, ctx) -> dict:
    prompt = judge_prompt(row, gold, ctx)
    for _ in range(3):
        res = chat([{"role": "user", "content": prompt}], model=STRONG_MODEL, max_tokens=300, json_mode=True)
        try:
            d = json.loads(re.sub(r"^```(json)?|```$", "", res["text"].strip()).strip())
        except json.JSONDecodeError:
            continue
        allowed = REFUSE_LABELS if row["should_refuse"] == "true" else ANSWER_LABELS
        if d.get("label") in allowed:
            return {"label": d["label"], "reason": d.get("reason", "")}
    return {"label": "", "reason": "судья не вернул корректную метку"}


def load_ctx(config):
    path = RESULTS / f"{config}_ctx.jsonl"
    return {c["id"]: c for c in map(json.loads, path.open(encoding="utf-8"))} if path.exists() else {}


def run_judge(config, dry_run, workers):
    rows = read_csv(RESULTS / f"{config}.csv")
    gold = {q["id"]: q for q in read_csv(QUESTIONS)}
    ctx = load_ctx(config)
    out = RESULTS / f"{config}_judged.csv"
    done = {r["id"]: r for r in read_csv(out) if r.get("label")} if out.exists() else {}
    todo = [r for r in rows if r["id"] not in done]

    p_in, p_out = model_price(STRONG_MODEL)
    n_in = sum(len(ENC.encode(judge_prompt(r, gold[r["id"]], ctx.get(r["id"])))) for r in todo)
    n_out = 120 * len(todo)
    print(f"{config}: к оценке {len(todo)} из {len(rows)}; {STRONG_MODEL}: ~{n_in:,} вх. / ~{n_out:,} вых. "
          f"(cl100k), оценка с x2: ${2 * (n_in * p_in + n_out * p_out) / 1e6:.4f}")
    if dry_run:
        return

    judged = dict(done)
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(judge_one, r, gold[r["id"]], ctx.get(r["id"])): r for r in todo}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"judge {config}"):
            r = futures[f]
            judged[r["id"]] = {**r, **f.result()}
    result = []
    for r in rows:
        # метка и обоснование — от судьи, остальные поля — из текущего файла прогона
        j = {**judged[r["id"]], **r}
        links = answer_links(j["answer"])
        found = set(j["found_urls"].split())
        j["has_link"] = int(bool(links))
        j["links_found"] = int(all(u in found for u in links)) if links and j["mode"] == "rag" else ""
        result.append(j)
    fields = list(rows[0].keys()) + ["label", "reason", "has_link", "links_found"]
    write_csv(out, result, fields)
    empty = [r["id"] for r in result if not r["label"]]
    print(f"сохранено -> {out}" + (f"; без метки: {empty} (перезапустите)" if empty else ""))


def share(rows, cond, where=lambda r: True):
    base = [r for r in rows if where(r)]
    return (sum(cond(r) for r in base) / len(base), len(base)) if base else (None, 0)


def metrics(rows) -> dict:
    answerable = lambda r: r["should_refuse"] == "false"
    m = {
        "n": len(rows),
        "correct": share(rows, lambda r: r["label"] == "верно", answerable),
        "partial": share(rows, lambda r: r["label"] == "частично", answerable),
        "refusal_ok": share(rows, lambda r: r["label"] == "корректный_отказ", lambda r: not answerable(r)),
        "hallucinated": share(rows, lambda r: r["label"] == "галлюцинация"),
        "link": share(rows, lambda r: r["has_link"] == "1", answerable),
        "link_found": share(rows, lambda r: r["links_found"] == "1", lambda r: r["links_found"] != ""),
        "hit5": share(rows, lambda r: r["hit5"] == "1", lambda r: r["hit5"] != ""),
        "mrr": (sum(float(r["rr"]) for r in rows if r["rr"] != "") / max(1, sum(r["rr"] != "" for r in rows)),
                sum(r["rr"] != "" for r in rows)),
        "latency": (sum(float(r["latency_s"]) for r in rows) / len(rows), len(rows)) if rows else (None, 0),
    }
    return m


def fmt(v, pct=True):
    val, n = v
    if val is None or n == 0:
        return "—"
    return f"{val:.0%}" if pct else f"{val:.2f}"


def report(configs):
    cols = [("correct", "верно"), ("partial", "частично"), ("refusal_ok", "корр. отказ"),
            ("hallucinated", "галлюц."), ("link", "ссылка"), ("link_found", "ссылки из найденных"),
            ("hit5", "hit@5"), ("mrr", "MRR"), ("latency", "время, с")]
    lines = ["| config | split | " + " | ".join(c[1] for c in cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    by_type = {}
    split_of = {q["id"]: q["split"] for q in read_csv(QUESTIONS)}
    for config in configs:
        rows = read_csv(RESULTS / f"{config}_judged.csv")
        for r in rows:
            r["split"] = split_of[r["id"]]  # источник истины для разбиения — questions.csv
        for split in ("dev", "holdout", "all"):
            part = [r for r in rows if split == "all" or r["split"] == split]
            m = metrics(part)
            lines.append(f"| {config} | {split} | " + " | ".join(
                fmt(m[k], pct=k not in ("mrr", "latency")) for k, _ in cols) + " |")
        for r in rows:
            by_type.setdefault(r["type"], {}).setdefault(config, []).append(r["label"])
    print("\n".join(lines))

    print("\nРазбивка по типам вопросов (все 40, метки судьи):")
    for t, per_config in by_type.items():
        print(f"\n{t}:")
        for config, labels in per_config.items():
            counts = {l: labels.count(l) for l in LABELS if labels.count(l)}
            print(f"  {config}: " + ", ".join(f"{l} {c}" for l, c in counts.items()))
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nтаблица -> {RESULTS / 'summary.md'}")


def sample_manual(configs):
    if MANUAL.exists() and any(r.get("human_label") for r in read_csv(MANUAL)):
        raise SystemExit(f"{MANUAL} уже заполнен вручную, не перезаписываю")
    pool = [r for c in configs for r in read_csv(RESULTS / f"{c}_judged.csv")]
    picked = random.Random(SEED).sample(pool, MANUAL_N)
    gold = {q["id"]: q for q in read_csv(QUESTIONS)}
    rows = [{"config": r["config"], "id": r["id"], "type": r["type"], "should_refuse": r["should_refuse"],
             "question": r["question"], "gold_answer": gold[r["id"]]["gold_answer"],
             "gold_urls": gold[r["id"]]["gold_urls"], "answer": r["answer"], "found_urls": r["found_urls"],
             "judge_label": r["label"], "judge_reason": r["reason"], "human_label": ""} for r in picked]
    write_csv(MANUAL, rows, list(rows[0].keys()))
    print(f"{MANUAL_N} строк -> {MANUAL}. Заполните столбец human_label одной из меток: {', '.join(LABELS)}")


def agreement():
    rows = [r for r in read_csv(MANUAL) if r["human_label"].strip()]
    if not rows:
        raise SystemExit(f"в {MANUAL} не заполнен human_label")
    bad = [r["human_label"] for r in rows if r["human_label"].strip() not in LABELS]
    if bad:
        raise SystemExit(f"неизвестные метки: {bad}")
    same = [r for r in rows if r["human_label"].strip() == r["judge_label"]]
    print(f"совпадений: {len(same)} из {len(rows)} ({len(same) / len(rows):.0%})")
    for r in rows:
        if r not in same:
            print(f"  {r['config']} {r['id']}: судья «{r['judge_label']}», человек «{r['human_label']}»")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", nargs="+")
    ap.add_argument("--sample-manual", nargs="+")
    ap.add_argument("--agreement", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    elif args.sample_manual:
        sample_manual(args.sample_manual)
    elif args.agreement:
        agreement()
    elif args.config:
        run_judge(args.config, args.dry_run, args.workers)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
