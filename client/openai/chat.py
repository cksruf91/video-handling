import os
from typing import Self

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_KEY_VIDEO"))
        self.model = "gpt-4o-mini"
        self.messages = []
        self._role_pool = ["system", "assistant", "user"]

    def add_prompt(self, role: str, text: str) -> Self:
        if role not in self._role_pool:
            raise ValueError(f"Role {role} not in {self._role_pool}")
        self.messages.append(
            {"role": role, "content": text},
        )
        return self

    def clear(self) -> Self:
        self.messages = []
        return self

    def call(self, **kwargs) -> str | ChatCompletion:
        _parsing = kwargs.pop('parsing') if 'parsing' in kwargs else True
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **kwargs
        )
        if _parsing:
            return completion.choices[0].message.content
        return completion
