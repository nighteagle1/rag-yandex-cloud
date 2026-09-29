from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.message import Message

SOURCES_MARKER = "\n\n---\nНайденные страницы:\n"


class FormatAnswer(Component):
    """Добавляет к ответу модели список URL, найденных поиском."""

    display_name = "Format Answer"
    description = "Ответ модели + список найденных страниц."
    icon = "list"
    name = "FormatAnswer"

    inputs = [
        MessageTextInput(name="answer", display_name="Answer", required=True),
        MessageTextInput(name="sources", display_name="Sources", required=True),
    ]

    outputs = [Output(display_name="Message", name="message", method="build_message")]

    def build_message(self) -> Message:
        urls = [u for u in self.sources.split("\n") if u.strip()]
        text = self.answer.strip() + SOURCES_MARKER + "\n".join(f"- {u}" for u in urls)
        return Message(text=text)
