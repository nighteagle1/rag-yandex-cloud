# CLAUDE.md

RAG-ассистент первой линии поддержки по русской документации Yandex Cloud (Compute Cloud, Object Storage, VPC, Billing). Поток собран в Langflow, трассировка в Langfuse Cloud, оценка качества на 40 вопросах. Полное задание — `plan.md`, требования и пороги метрик — `docs/requirements.md`, текущий статус — `project_status.md`.

## Правила работы
- Общайся с пользователем по-русски.
- Коммить после каждого шага плана, окончание сообщения коммита — строка `Co-Authored-By`.
- **Бюджет OpenRouter: $0.59 на весь проект.** На ключе до старта уже было потрачено $1.394872396 (`data/budget.json`). Расход проекта: `common.spent_usd() - baseline`. Перед каждым массовым платным вызовом оценивай стоимость (`--dry-run`, `common.model_price`).
- Не выдумывай числа: всё, что попадает в README, берётся из результатов прогонов.
- `.env` не читать и не печатать; ключи только в `.env` (он в `.gitignore`).
- Holdout-вопросы (`split=holdout`) не смотреть при правке промптов.

## Окружение (Windows 10, Python 3.13, Docker)
- `docker compose up -d` поднимает Qdrant 1.15.4 (`localhost:6333`, **именованный том** `qdrant_data`) и Langflow 1.12.3 (`localhost:7860`, без логина, только localhost).
- **Не пересоздавай контейнер Qdrant во время загрузки** и не возвращай bind-mount: на Windows хранилище портится (было: `OutputTooSmall` и потерянные векторы). Чтобы перезапустить только Langflow: `docker compose up -d --no-deps langflow`.
- Скрипты запускай из `scripts/` с `PYTHONIOENCODING=utf-8` (консоль cp866).
- В Bash-heredoc не пиши Python-строки с `\n` и `\1`: они ломаются. Для правок используй Edit.

## Модели (OpenRouter, заданы в `scripts/common.py`, переопределяются через env)
- Ответы: `qwen/qwen3-30b-a3b-instruct-2507` (`ANSWER_MODEL`).
- Вопросы и судья: `deepseek/deepseek-v4-flash-0731` (`STRONG_MODEL`), reasoning отключается через `extra_body.reasoning.enabled=false`.
- Эмбеддинги: `qwen/qwen3-embedding-8b`, размерность 4096 (`EMBED_MODEL`). Запросы эмбеддятся с префиксом `common.QUERY_INSTRUCTION`, документы — без него.

## Скрипты
- `prepare_corpus.py` — `yc-docs/ru` → `data/corpus.jsonl` (453 страницы). Разворачивает include, подставляет presets, ссылки `{#T}`, чистит YFM-таблицы.
- `check_embeddings.py [model...]` — тест модели эмбеддингов на 5 русских парах.
- `ingest.py --chunk-size N --collection C [--dry-run|--yes|--query Q]` — чанки по заголовкам, затем по размеру с перекрытием 100 токенов (cl100k). Кэш векторов лежит в `data/emb_cache/` (нулевые векторы пересчитываются), поэтому повторная индексация бесплатна.
- `build_flow.py [--prompt v2] [--collection yc_c1000]` — собирает поток через REST API, пересоздаёт его и экспортирует в `flows/answer.json`, id записывает в `flows/flow_id.txt`.
- `common.py` — клиенты, `embed`, `chat`, `search`, `spent_usd`, `model_price`.

## Langflow: особенности версии 1.12.3
- Компонента Qdrant в образе нет (вынесен в пакет, которого нет на PyPI), отдельного компонента OpenRouter тоже нет. Поэтому используются пользовательские компоненты `flows/components/qdrant_search.py` и `format_answer.py`, которые создаются через `POST /api/v1/custom_component`.
- Эмбеддинги и LLM — компоненты OpenAI с `openai_api_base=https://openrouter.ai/api/v1`, ключ берётся из глобальной переменной `OPENROUTER_API_KEY` (`load_from_db`). Для этого нужны env `LANGFLOW_PROVIDER_CREDENTIAL_ALLOWED_HOSTS=openrouter.ai` и `LANGFLOW_VARIABLES_TO_GET_FROM_ENVIRONMENT`.
- В эмбеддингах `tiktoken_enable=False`: иначе langchain шлёт токены cl100k, которые qwen не понимает.
- Фиксированные id узлов для `tweaks` лежат в `build_flow.IDS`, например `QdrantSearch-yc003.collection_name` и `Prompt-yc004.template`.
- Ответ потока: текст модели, затем маркер `\n\n---\nНайденные страницы:\n` и список URL.
- Langfuse подключается переменными `LANGFUSE_*` из `.env` (через `env_file`).
