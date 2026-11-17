"""Chat models behind a small interface, with streaming support."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class ChatModel(Protocol):
    def complete(self, messages: list[dict]) -> str: ...

    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class OpenAIChat:
    def __init__(
        self, model: str = "gpt-4o-mini", api_key: str | None = None, temperature: float = 0.0
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        return response.choices[0].message.content or ""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, stream=True
        )
        for event in response:
            if event.choices and (delta := event.choices[0].delta.content):
                yield delta
