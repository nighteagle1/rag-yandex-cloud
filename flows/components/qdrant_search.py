from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, IntInput, MessageTextInput, MultilineInput, Output, StrInput
from lfx.schema.message import Message
from qdrant_client import QdrantClient


class QdrantSearch(Component):
    """Поиск top_k чанков в коллекции Qdrant по эмбеддингу вопроса.

    Компонент Qdrant вынесен из Langflow 1.12 в отдельный пакет, которого нет в образе,
    поэтому поиск реализован здесь напрямую через qdrant-client.
    """

    display_name = "Qdrant Search"
    description = "Ищет ближайшие чанки в коллекции Qdrant (cosine)."
    icon = "database"
    name = "QdrantSearch"

    inputs = [
        MessageTextInput(name="query", display_name="Query", required=True),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"], required=True),
        StrInput(name="collection_name", display_name="Collection", value="yc_c1000"),
        StrInput(name="qdrant_url", display_name="Qdrant URL", value="http://qdrant:6333"),
        IntInput(name="top_k", display_name="Top K", value=5),
        MultilineInput(name="query_instruction", display_name="Query Instruction", value="", advanced=True),
    ]

    # group_outputs=True: оба выхода видны в интерфейсе одновременно; без этого фронтенд Langflow
    # показывает только выбранный выход и при сохранении потока удаляет связь от второго
    outputs = [
        Output(display_name="Context", name="context", method="build_context", group_outputs=True),
        Output(display_name="Sources", name="sources", method="build_sources", group_outputs=True),
    ]

    def _hits(self):
        if getattr(self, "_cached_hits", None) is None:
            # тот же префикс, что в common.QUERY_INSTRUCTION: документы индексировались без него
            vector = self.embedding.embed_query(self.query_instruction + self.query)
            client = QdrantClient(url=self.qdrant_url, timeout=60)
            self._cached_hits = client.query_points(
                collection_name=self.collection_name, query=vector, limit=self.top_k, with_payload=True
            ).points
        return self._cached_hits

    def build_context(self) -> Message:
        parts = []
        for i, h in enumerate(self._hits(), 1):
            p = h.payload
            parts.append(f"[{i}] URL: {p['url']}\n{p['text']}")
        text = "\n\n---\n\n".join(parts)
        self.status = f"{len(parts)} чанков из {self.collection_name}"
        return Message(text=text)

    def build_sources(self) -> Message:
        urls = []
        for h in self._hits():
            if h.payload["url"] not in urls:
                urls.append(h.payload["url"])
        return Message(text="\n".join(urls))
