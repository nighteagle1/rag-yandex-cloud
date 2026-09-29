"""Готовит корпус из русской документации yandex-cloud/docs.

Читает .md выбранных разделов, разворачивает {% include %}, подставляет
переменные {{ ... }} из presets.yaml, убирает остальную разметку YFM
и сохраняет страницы в data/corpus.jsonl (title, service, url, path, text).

    python scripts/prepare_corpus.py --docs yc-docs
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# service -> разделы внутри ru/<service>/
SECTIONS = {
    "compute": ["concepts", "operations", "qa", "quickstart", "pricing.md"],
    "storage": ["concepts", "operations", "quickstart", "quickstart.md", "qa.md", "pricing.md"],
    "vpc": ["concepts", "operations", "qa", "quickstart.md", "pricing.md"],
    "billing": ["concepts", "operations", "qa", "quickstart", "payment", "usage", "pricing.md"],
}
SERVICE_NAMES = {"compute": "Compute Cloud", "storage": "Object Storage", "vpc": "VPC", "billing": "Billing"}
BASE_URL = "https://yandex.cloud/ru/docs/"

FRONT_MATTER = re.compile(r"\A﻿?---\n.*?\n---\n", re.S)
INCLUDE = re.compile(r"\{%-?\s*include\s+(?:notitle\s+)?\[[^\]]*\]\(([^)]+)\)\s*-?%\}")
VAR = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")
ANCHOR = re.compile(r"\s*\{#[^}]*\}")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = "" if v is None else str(v)
    return out


def load_presets(ru: Path, service: str) -> dict:
    presets = flatten(yaml.safe_load((ru / "presets.yaml").read_text(encoding="utf-8"))["default"])
    local = ru / service / "presets.yaml"
    if local.exists():
        data = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        presets.update(flatten(data.get("default", {})))
    return presets


def extract_section(text: str, anchor: str) -> str:
    """Фрагмент от заголовка с {#anchor} до следующего заголовка того же или более высокого уровня."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(#+)\s", line)
        if m and "{#" + anchor + "}" in line:
            level = len(m.group(1))
            end = len(lines)
            for j in range(i + 1, len(lines)):
                m2 = re.match(r"^(#+)\s", lines[j])
                if m2 and len(m2.group(1)) <= level:
                    end = j
                    break
            return "\n".join(lines[i + 1:end])
    return ""


def target_title(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    fm = FRONT_MATTER.match(raw)
    if fm:
        meta = yaml.safe_load(fm.group(0).strip("﻿-\n")) or {}
        if meta.get("title"):
            return str(meta["title"])
    h1 = re.search(r"^# (.+)$", raw, re.M)
    return ANCHOR.sub("", h1.group(1)).strip() if h1 else ""


def resolve_auto_links(text: str, base: Path) -> str:
    """[{#T}](page.md) — YFM подставляет заголовок целевой страницы; делаем то же."""
    def repl(m):
        target = m.group(1).split("#")[0]
        title = target_title((base.parent / target).resolve()) if target.endswith(".md") else ""
        return title or ""

    return re.sub(r"\[(?:\{#T\})?\]\((?!https?:)([^)\s]*)[^)]*\)", repl, text)


def expand_includes(text: str, base: Path, stats: dict, depth=0) -> str:
    text = resolve_auto_links(text, base)

    def repl(m):
        target = m.group(1).strip()
        path_part, _, anchor = target.partition("#")
        path = (base.parent / path_part).resolve()
        if depth > 10 or not path.exists():
            stats["missing_include"] += 1
            return ""
        inc = FRONT_MATTER.sub("", path.read_text(encoding="utf-8"))
        if anchor:
            inc = extract_section(inc, anchor)
        return expand_includes(inc, path, stats, depth + 1)

    return INCLUDE.sub(repl, text)


def substitute_vars(text: str, presets: dict, stats: dict) -> str:
    def repl(m):
        key = m.group(1)
        if key in presets:
            return presets[key]
        stats["unknown_vars"].add(key)
        return ""

    for _ in range(3):  # значения пресетов сами могут содержать переменные
        text = VAR.sub(repl, text)
    return text


def convert_tabs(text: str) -> str:
    """{% list tabs %} -> пункты верхнего уровня становятся подзаголовками вкладок."""
    out, in_tabs = [], False
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"\{%-?\s*list tabs", s):
            in_tabs = True
            continue
        if re.match(r"\{%-?\s*endlist", s):
            in_tabs = False
            continue
        if in_tabs:
            m = re.match(r"^- (.+)$", line)
            if m:
                out.append(f"**{m.group(1).strip()}:**")
                continue
            if line.startswith("  "):
                line = line[2:]
        out.append(line)
    return "\n".join(out)


def clean(text: str) -> str:
    text = re.sub(r"::: *page-constructor.*?\n:::", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # цены подставляются из прайс-листа при сборке сайта, в репозитории их нет
    text = re.sub(r"<MDX>.*?</MDX>", "(актуальные цены — в прайс-листе на сайте)", text, flags=re.S)
    text = re.sub(r"\{%-?\s*calc\b[^%]*%\}", "<итог>", text)
    text = re.sub(r"\{\{\s*sku\|[A-Z]+\|([^|}]+)\|[^}]*\}\}", r"<цена \1>", text)
    text = convert_tabs(text)
    text = re.sub(r"\{%-?\s*cut\s+\"([^\"]*)\"\s*-?%\}", r"**\1**", text)
    text = re.sub(r"\{%-?\s*note\s+\w+\s+\"([^\"]*)\"\s*-?%\}", r"\1:", text)
    text = re.sub(r"\{%-?\s*note\s+(info|tip)\s*-?%\}", "Примечание:", text)
    text = re.sub(r"\{%-?\s*note\s+(warning|alert)\s*-?%\}", "Важно:", text)
    text = re.sub(r"\{%-?\s*note\s*[^%]*%\}", "", text)
    text = re.sub(r"\{%-?\s*[^%]*?-?%\}", "", text)  # endnote, endcut, if/endif и прочие теги
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # картинки
    text = re.sub(r"\[([^\]]*)\]\((?!https?:)[^)]*\)", r"\1", text)  # относительные ссылки -> текст
    text = re.sub(r"\[([^\]]*)\]\((https?:[^)\s]*)[^)]*\)", r"\1 (\2)", text)
    text = ANCHOR.sub("", text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<sup>([^<]*)</sup>", r"^\1", text)
    text = re.sub(r"</?q>", "\"", text)
    text = re.sub(r"</?code>", "`", text)
    text = re.sub(r"<li>", "\n- ", text)
    text = re.sub(r"</?(?:small|sup|sub|b|i|p|span|div|ul|ol|li)[^>]*>", "", text)
    # YFM-таблицы #| || ячейка | ячейка || |#
    text = re.sub(r"^\s*(#\||\|#)\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\|\|\s*", "", text, flags=re.M)
    text = re.sub(r"\s*\|\|\s*$", "", text, flags=re.M)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_url(rel: Path) -> str:
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "index":
        parts = parts[:-1]
    return BASE_URL + "/".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "yc-docs"))
    ap.add_argument("--out", default=str(ROOT / "data" / "corpus.jsonl"))
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--sample", type=int, default=0, help="вывести N случайных очищенных страниц")
    args = ap.parse_args()

    ru = Path(args.docs) / "ru"
    stats = {"missing_include": 0, "unknown_vars": set(), "skipped_short": 0}
    pages = []
    for service, sections in SECTIONS.items():
        presets = load_presets(ru, service)
        files = []
        for sec in sections:
            p = ru / service / sec
            files += [p] if p.is_file() else sorted(p.rglob("*.md")) if p.exists() else []
        files = [f for f in files if "api-ref" not in f.parts]
        for f in files:
            raw = f.read_text(encoding="utf-8")
            fm = FRONT_MATTER.match(raw)
            meta = yaml.safe_load(fm.group(0).strip("﻿-\n")) if fm else {}
            body = FRONT_MATTER.sub("", raw)
            body = expand_includes(body, f, stats)
            body = substitute_vars(body, presets, stats)
            text = clean(body)
            h1 = re.search(r"^# (.+)$", text, re.M)
            title = h1.group(1).strip() if h1 else substitute_vars(str((meta or {}).get("title", f.stem)), presets, stats)
            if len(text) < args.min_chars:
                stats["skipped_short"] += 1
                continue
            rel = f.relative_to(ru)
            pages.append({"title": title, "service": SERVICE_NAMES[service], "url": page_url(rel),
                          "path": rel.as_posix(), "text": text})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = sum(len(enc.encode(p["text"])) for p in pages)
    except ImportError:
        tokens = None
    by_service = {}
    for p in pages:
        by_service[p["service"]] = by_service.get(p["service"], 0) + 1
    print(f"pages: {len(pages)} {by_service}")
    print(f"tokens (cl100k): {tokens}")
    print(f"skipped short: {stats['skipped_short']}, missing includes: {stats['missing_include']}")
    print(f"unknown vars ({len(stats['unknown_vars'])}): {sorted(stats['unknown_vars'])[:20]}")
    leftovers = sum(bool(re.search(r"\{%|%\}|\{\{|\}\}|:::", p["text"])) for p in pages)
    print(f"pages with markup leftovers: {leftovers}")

    if args.sample:
        random.seed(0)
        for p in random.sample(pages, args.sample):
            sys.stdout.write(f"\n===== {p['url']} | {p['title']}\n{p['text'][:1200]}\n")


if __name__ == "__main__":
    main()
