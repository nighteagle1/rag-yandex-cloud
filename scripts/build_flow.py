"""Собирает поток RAG в Langflow через REST API и экспортирует его в flows/answer.json.

ChatInput -> QdrantSearch (+ OpenAI Embeddings с base URL OpenRouter) -> Prompt Template
-> OpenAI Model (base URL OpenRouter, t=0) -> FormatAnswer (+ список URL) -> ChatOutput

    python scripts/build_flow.py [--prompt v2] [--collection yc_c1000]
"""
import argparse
import copy
import json
import os

import requests

from common import ANSWER_MODEL, EMBED_MODEL, OPENROUTER_BASE_URL, QUERY_INSTRUCTION, ROOT, load_prompt

LF = os.getenv("LANGFLOW_URL", "http://localhost:7860").rstrip("/")
FLOW_NAME = "yc-support-rag"
KEY_VAR = "OPENROUTER_API_KEY"  # глобальная переменная Langflow, берётся из окружения контейнера

# фиксированные id узлов: на них ссылаются tweaks в run_eval.py
IDS = {
    "input": "ChatInput-yc001",
    "embed": "OpenAIEmbeddings-yc002",
    "search": "QdrantSearch-yc003",
    "prompt": "Prompt-yc004",
    "llm": "OpenAIModel-yc005",
    "format": "FormatAnswer-yc006",
    "output": "ChatOutput-yc007",
}


def api(method, path, **kw):
    r = requests.request(method, f"{LF}/api/v1{path}", timeout=120, **kw)
    if r.status_code >= 400:
        raise SystemExit(f"{method} {path}: {r.status_code} {r.text[:500]}")
    return r.json()


def template_node(all_types, category, type_name):
    return copy.deepcopy(all_types[category][type_name])


def custom_node(path):
    code = (ROOT / "flows" / "components" / path).read_text(encoding="utf-8")
    return api("POST", "/custom_component", json={"code": code})["data"]


def set_field(node, field, value, load_from_db=False):
    node["template"][field]["value"] = value
    if load_from_db:
        node["template"][field]["load_from_db"] = True


def wrap(node_id, type_name, node, x, y):
    return {"id": node_id, "type": "genericNode", "position": {"x": x, "y": y},
            "data": {"id": node_id, "type": type_name, "node": node}}


def handle_str(d):
    # формат строковых хэндлов фронтенда Langflow: кавычки заменены на «œ»
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False).replace('"', "œ")


def edge(nodes, src, out_name, tgt, field):
    s, t = nodes[src]["data"], nodes[tgt]["data"]
    out = next(o for o in s["node"]["outputs"] if o["name"] == out_name)
    fld = t["node"]["template"][field]
    source_handle = {"dataType": s["type"], "id": s["id"], "name": out_name, "output_types": out["types"]}
    target_handle = {"fieldName": field, "id": t["id"], "inputTypes": fld.get("input_types") or [],
                     "type": fld.get("type", "str")}
    sh, th = handle_str(source_handle), handle_str(target_handle)
    return {"id": f"xy-edge__{s['id']}{sh}-{t['id']}{th}", "source": s["id"], "target": t["id"],
            "sourceHandle": sh, "targetHandle": th,
            "data": {"sourceHandle": source_handle, "targetHandle": target_handle}}


def prompt_node(all_types, template_text):
    node = template_node(all_types, "models_and_agents", "Prompt Template")
    set_field(node, "template", template_text)
    variables = ["context", "question"]
    for var in variables:
        node["template"][var] = {
            "field_type": "str", "type": "str", "name": var, "display_name": var, "value": "",
            "required": False, "placeholder": "", "show": True, "advanced": False, "multiline": True,
            "input_types": ["Message"], "list": False, "load_from_db": False, "dynamic": False,
            "info": "", "title_case": False, "_input_type": "MessageTextInput",
        }
    node["custom_fields"] = {"template": variables}
    return node


def build(prompt_name, collection):
    all_types = requests.get(f"{LF}/api/v1/all", timeout=120).json()

    chat_in = template_node(all_types, "input_output", "ChatInput")

    emb = template_node(all_types, "openai", "ext:openai:OpenAIEmbeddingsComponent@official")
    set_field(emb, "model", EMBED_MODEL)
    # как и у LLM: иначе Langflow сбрасывает значение не из списка моделей OpenAI в пустую строку
    emb["template"]["model"]["combobox"] = True
    emb["template"]["model"]["options"] = [EMBED_MODEL]
    set_field(emb, "openai_api_base", OPENROUTER_BASE_URL)
    set_field(emb, "openai_api_key", KEY_VAR, load_from_db=True)
    # без tiktoken langchain отправляет текст, а не токены cl100k (qwen их не понимает)
    set_field(emb, "tiktoken_enable", False)
    set_field(emb, "tiktoken_model_name", "Qwen/Qwen3-Embedding-8B")
    set_field(emb, "embedding_ctx_length", 8000)

    search = custom_node("qdrant_search.py")
    set_field(search, "collection_name", collection)
    set_field(search, "top_k", 5)
    set_field(search, "query_instruction", QUERY_INSTRUCTION)

    prompt = prompt_node(all_types, load_prompt(prompt_name))

    llm = template_node(all_types, "openai", "ext:openai:OpenAIModelComponent@official")
    set_field(llm, "model_name", ANSWER_MODEL)
    set_field(llm, "openai_api_base", OPENROUTER_BASE_URL)
    set_field(llm, "api_key", KEY_VAR, load_from_db=True)
    set_field(llm, "temperature", 0.0)
    set_field(llm, "max_tokens", 800)
    # модель задаётся строкой, а не из списка моделей OpenAI
    llm["template"]["model_name"]["combobox"] = True
    llm["template"]["model_name"]["options"] = [ANSWER_MODEL]

    fmt = custom_node("format_answer.py")
    chat_out = template_node(all_types, "input_output", "ChatOutput")

    nodes = {
        "input": wrap(IDS["input"], "ChatInput", chat_in, 0, 200),
        "embed": wrap(IDS["embed"], "OpenAIEmbeddings", emb, 0, 500),
        "search": wrap(IDS["search"], "QdrantSearch", search, 400, 300),
        "prompt": wrap(IDS["prompt"], "Prompt", prompt, 800, 200),
        "llm": wrap(IDS["llm"], "OpenAIModel", llm, 1200, 200),
        "format": wrap(IDS["format"], "FormatAnswer", fmt, 1600, 300),
        "output": wrap(IDS["output"], "ChatOutput", chat_out, 2000, 300),
    }
    edges = [
        edge(nodes, "input", "message", "search", "query"),
        edge(nodes, "embed", "embeddings", "search", "embedding"),
        edge(nodes, "search", "context", "prompt", "context"),
        edge(nodes, "input", "message", "prompt", "question"),
        edge(nodes, "prompt", "prompt", "llm", "input_value"),
        edge(nodes, "llm", "text_output", "format", "answer"),
        edge(nodes, "search", "sources", "format", "sources"),
        edge(nodes, "format", "message", "output", "input_value"),
    ]
    return {"name": FLOW_NAME, "description": "RAG по документации Yandex Cloud",
            "data": {"nodes": list(nodes.values()), "edges": edges,
                     "viewport": {"x": 0, "y": 0, "zoom": 0.6}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v2")
    ap.add_argument("--collection", default="yc_c1000")
    args = ap.parse_args()

    flow = build(args.prompt, args.collection)
    for f in api("GET", "/flows/", params={"remove_example_flows": True, "get_all": True}):
        if f.get("name") == FLOW_NAME:
            api("DELETE", f"/flows/{f['id']}")
    created = api("POST", "/flows/", json=flow)
    flow_id = created["id"]
    exported = api("GET", f"/flows/{flow_id}")
    for k in ("user_id", "folder_id", "updated_at"):
        exported.pop(k, None)
    out = ROOT / "flows" / "answer.json"
    out.write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "flows" / "flow_id.txt").write_text(flow_id, encoding="utf-8")
    print(f"flow id: {flow_id}\nexported: {out}")


if __name__ == "__main__":
    main()
