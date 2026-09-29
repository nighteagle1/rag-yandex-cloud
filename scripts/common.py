"""Общие клиенты и утилиты: .env, OpenRouter (чат и эмбеддинги), Qdrant, корпус."""
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Модели выбраны по актуальному списку OpenRouter (см. README, раздел «Модели»)
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
STRONG_MODEL = os.getenv("STRONG_MODEL", "deepseek/deepseek-v4-flash-0731")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen/qwen3-embedding-8b")

CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
PROMPTS_DIR = ROOT / "prompts"


@lru_cache
def openrouter() -> OpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY не задан в .env")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, max_retries=5, timeout=120)


@lru_cache
def qdrant() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                        api_key=os.getenv("QDRANT_API_KEY") or None, timeout=60, check_compatibility=False)


def embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(5):
            try:
                resp = openrouter().embeddings.create(model=EMBED_MODEL, input=batch)
                vecs = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
                if len(vecs) != len(batch) or any(not any(v) for v in vecs):
                    raise ValueError("пустые или нулевые эмбеддинги в ответе")
                out += vecs
                break
            except Exception as e:  # сетевые ошибки и 429 — повторяем
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                print(f"embed retry {attempt + 1}: {e}")
    return out


def chat(messages: list[dict], model: str = ANSWER_MODEL, temperature: float = 0.0,
         max_tokens: int = 800, reasoning_off: bool = True, json_mode: bool = False) -> dict:
    """Вызов Chat Completions. Возвращает текст, usage и id генерации OpenRouter."""
    extra = {}
    if reasoning_off:
        extra["reasoning"] = {"enabled": False}
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                  extra_body=extra or None)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = openrouter().chat.completions.create(**kwargs)
    usage = resp.usage.model_dump() if resp.usage else {}
    return {"text": (resp.choices[0].message.content or "").strip(), "usage": usage, "id": resp.id,
            "model": resp.model}


def spent_usd() -> float | None:
    """Сколько потрачено ключом OpenRouter за всё время (по /api/v1/key)."""
    try:
        r = requests.get(f"{OPENROUTER_BASE_URL}/key",
                         headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}, timeout=30)
        return float(r.json()["data"]["usage"])
    except Exception:
        return None


def model_price(model: str) -> tuple[float, float]:
    """Цена за 1M токенов (вход, выход) по публичному списку моделей OpenRouter."""
    for url in (f"{OPENROUTER_BASE_URL}/models", f"{OPENROUTER_BASE_URL}/embeddings/models"):
        data = requests.get(url, timeout=30).json()["data"]
        for m in data:
            if m["id"] == model:
                p = m["pricing"]
                return float(p.get("prompt", 0)) * 1e6, float(p.get("completion", 0) or 0) * 1e6
    raise ValueError(f"модель {model} не найдена в OpenRouter")


def load_corpus() -> list[dict]:
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


# Qwen3-Embedding рекомендует инструкцию для запросов (документы индексируются без неё)
QUERY_INSTRUCTION = ("Instruct: Given a question from a cloud support engineer, "
                     "retrieve documentation passages that answer the question\nQuery: ")


def search(query: str, collection: str, top_k: int = 5) -> list[dict]:
    vec = embed([QUERY_INSTRUCTION + query])[0]
    hits = qdrant().query_points(collection_name=collection, query=vec, limit=top_k, with_payload=True).points
    return [{"score": h.score, **h.payload} for h in hits]


LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://localhost:7860").rstrip("/")
SOURCES_MARKER = "\n\n---\nНайденные страницы:\n"  # как в flows/components/format_answer.py


def run_flow(question: str, collection: str | None = None, prompt_text: str | None = None,
             session_id: str | None = None) -> dict:
    """Вызов потока Langflow (POST /api/v1/run). Коллекция и промпт переопределяются через tweaks.

    Возвращает ответ модели без списка источников, найденные URL, id трейса Langfuse и время.
    """
    from build_flow import IDS  # ленивый импорт: build_flow сам импортирует common

    flow_id = (ROOT / "flows" / "flow_id.txt").read_text(encoding="utf-8").strip()
    # модели передаются явно: Langflow может сбросить значение не из своего списка моделей OpenAI
    tweaks = {IDS["search"]: {}, IDS["prompt"]: {}, IDS["embed"]: {"model": EMBED_MODEL},
              IDS["llm"]: {"model_name": ANSWER_MODEL}}
    if collection:
        tweaks[IDS["search"]]["collection_name"] = collection
    if prompt_text:
        tweaks[IDS["prompt"]]["template"] = prompt_text
    tweaks = {k: v for k, v in tweaks.items() if v}
    body = {"input_value": question, "input_type": "chat", "output_type": "chat", "tweaks": tweaks}
    if session_id:
        body["session_id"] = session_id
    t0 = time.time()
    r = requests.post(f"{LANGFLOW_URL}/api/v1/run/{flow_id}", json=body, timeout=300)
    r.raise_for_status()
    msg = r.json()["outputs"][0]["outputs"][0]["results"]["message"]
    text = msg["text"]
    answer, _, sources = text.partition(SOURCES_MARKER)
    urls = [u.strip().removeprefix("- ").strip() for u in sources.splitlines() if u.strip()]
    meta = msg.get("session_metadata") or msg.get("data", {}).get("session_metadata") or {}
    return {"answer": answer.strip(), "urls": urls, "raw": text, "trace_id": meta.get("langfuse_trace_id"),
            "run_id": msg.get("run_id"), "latency_s": round(time.time() - t0, 2)}
