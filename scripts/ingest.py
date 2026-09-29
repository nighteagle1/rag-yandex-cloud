"""Разбивает корпус на чанки и индексирует их в Qdrant.

    python scripts/ingest.py --chunk-size 1000 --collection yc_c1000 --dry-run
    python scripts/ingest.py --chunk-size 1000 --collection yc_c1000 --yes
    python scripts/ingest.py --collection yc_c1000 --query "как увеличить квоту на vCPU"
"""
import argparse
import json
import re
import time
import uuid

import numpy as np
import tiktoken
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from common import EMBED_MODEL, ROOT, embed, load_corpus, model_price, qdrant, search

CACHE_DIR = ROOT / "data" / "emb_cache"

ENC = tiktoken.get_encoding("cl100k_base")
OVERLAP = 100
HEADING = re.compile(r"^(#{1,4})\s+(.+)$")


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """[(путь заголовков, текст секции)]; заголовки внутри ``` не считаются."""
    sections, path, buf, in_code = [], [], [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            if "".join(buf).strip():
                sections.append((" / ".join(path), "\n".join(buf).strip()))
            level = len(m.group(1))
            path = path[:level - 1] + [m.group(2).strip()]
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((" / ".join(path), "\n".join(buf).strip()))
    return sections


def split_by_size(text: str, size: int, overlap: int = OVERLAP) -> list[str]:
    tokens = ENC.encode(text)
    if len(tokens) <= size:
        return [text]
    parts, step = [], size - overlap
    for start in range(0, len(tokens), step):
        parts.append(ENC.decode(tokens[start:start + size]))
        if start + size >= len(tokens):
            break
    return parts


def make_chunks(pages: list[dict], size: int) -> list[dict]:
    chunks = []
    for page in pages:
        idx = 0
        # соседние короткие секции склеиваем, чтобы не плодить крошечные чанки
        merged, cur_head, cur = [], None, ""
        for head, body in split_by_headings(page["text"]):
            block = (f"## {head}\n" if head else "") + body
            if cur and len(ENC.encode(cur + "\n\n" + block)) > size:
                merged.append(cur)
                cur = block
            else:
                cur = (cur + "\n\n" + block) if cur else block
        if cur:
            merged.append(cur)
        for block in merged:
            for part in split_by_size(block, size):
                text = f"{page['title']}\n\n{part}"
                chunks.append({"url": page["url"], "title": page["title"], "service": page["service"],
                               "chunk_index": idx, "text": text})
                idx += 1
    return chunks


def point_id(chunk: dict) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['url']}#{chunk['chunk_index']}"))


def load_cache(collection: str) -> dict:
    """Кэш векторов по id точки: повторная индексация не платит за уже посчитанные эмбеддинги."""
    ids_f, vecs_f = CACHE_DIR / f"{collection}_ids.json", CACHE_DIR / f"{collection}_vecs.npy"
    if not ids_f.exists():
        return {}
    ids = json.loads(ids_f.read_text())
    # нулевые векторы — повреждённые записи, их пересчитываем
    return {i: v for i, v in zip(ids, np.load(vecs_f)) if np.linalg.norm(v) > 1e-6}


def save_cache(collection: str, cache: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids = list(cache)
    (CACHE_DIR / f"{collection}_ids.json").write_text(json.dumps(ids))
    np.save(CACHE_DIR / f"{collection}_vecs.npy", np.array([cache[i] for i in ids], dtype=np.float32))


def upsert(client, collection, points):
    for attempt in range(5):
        try:
            return client.upsert(collection, points=points, wait=True)
        except Exception as e:
            if attempt == 4:
                raise
            print(f"upsert retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--dry-run", action="store_true", help="только оценка числа токенов и стоимости")
    ap.add_argument("--yes", action="store_true", help="не спрашивать подтверждение")
    ap.add_argument("--query", action="append", help="проверочный поиск по коллекции")
    args = ap.parse_args()

    if args.query:
        for q in args.query:
            print(f"\nQ: {q}")
            for h in search(q, args.collection):
                print(f"  {h['score']:.3f}  {h['url']}  #{h['chunk_index']}")
        return

    chunks = make_chunks(load_corpus(), args.chunk_size)
    cache = load_cache(args.collection)
    todo = [c for c in chunks if point_id(c) not in cache]
    n_tokens = sum(len(ENC.encode(c["text"])) for c in todo)
    price_in, _ = model_price(EMBED_MODEL)
    cost = n_tokens / 1e6 * price_in
    print(f"chunks: {len(chunks)} (cached: {len(chunks) - len(todo)}), to embed: {len(todo)}, "
          f"tokens (cl100k): {n_tokens:,}, model: {EMBED_MODEL}, price ${price_in:.3f}/1M "
          f"-> estimated cost ${cost:.4f}")
    if args.dry_run:
        return
    if not args.yes and input("продолжить? [y/N] ").lower() != "y":
        return

    batch = 64
    try:
        for i in tqdm(range(0, len(todo), batch), desc="embed"):
            part = todo[i:i + batch]
            for c, v in zip(part, embed([c["text"] for c in part], batch_size=batch)):
                cache[point_id(c)] = np.asarray(v, dtype=np.float32)
    finally:
        save_cache(args.collection, cache)

    dim = len(next(iter(cache.values())))
    client = qdrant()
    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    client.create_collection(args.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    print(f"collection {args.collection}: dim={dim}, cosine")
    for i in tqdm(range(0, len(chunks), batch), desc="upsert"):
        part = chunks[i:i + batch]
        upsert(client, args.collection,
               [PointStruct(id=point_id(c), vector=cache[point_id(c)].tolist(), payload=c) for c in part])
    print(f"points in {args.collection}: {client.count(args.collection).count}")


if __name__ == "__main__":
    main()
