# RAG-ассистент по документации Yandex Cloud

RAG-ассистент для первой линии поддержки: отвечает по русской документации Yandex Cloud (Compute Cloud, Object Storage, VPC, Billing) со ссылками на страницы. Поток собран в Langflow, трейсы идут в Langfuse, качество измеряется воспроизводимым прогоном на 40 вопросах.

Требования и пороги метрик: [docs/requirements.md](docs/requirements.md).

## Корпус

- Источник: [yandex-cloud/docs](https://github.com/yandex-cloud/docs), клон `--depth 1`, коммит **`b54e74c8dd2e49a0355494a9a539be8ac1eca7e7`**.
- Разделы: `ru/{compute,storage,vpc,billing}/` — concepts, operations, qa, quickstart, pricing (для billing ещё payment и usage). Без `api-ref`, `cli-ref`, tutorials и release notes.
- Страниц: **453** (Compute Cloud 249, Object Storage 76, VPC 63, Billing 65).
- Токенов: **1 359 866** (оценка по `cl100k_base`).
- Очистка (`scripts/prepare_corpus.py`): удаляется front matter; `{% include %}` разворачиваются рекурсивно, в том числе фрагменты по якорю; `{{ переменные }}` подставляются из `presets.yaml`; ссылки `[{#T}](page.md)` заменяются заголовком целевой страницы; вкладки `{% list tabs %}` превращаются в подзаголовки; `note` и `cut` раскрываются; YFM-таблицы `#| … |#` приводятся к строкам.
- URL строятся из пути файла: `ru/compute/concepts/limits.md` → `https://yandex.cloud/ru/docs/compute/concepts/limits`. Три URL проверены вручную, страницы открываются.
- **Ограничение:** конкретных цен в репозитории нет. Они подставляются из прайс-листа при сборке сайта (`{{ sku|RUB|… }}`, `<PriceList>`), поэтому в корпусе вместо них стоят заглушки `<цена …>`. Вопросы «с числами» строятся на лимитах, размерах и сроках.

Проверка:
```bash
python scripts/prepare_corpus.py --docs yc-docs --sample 15
```
