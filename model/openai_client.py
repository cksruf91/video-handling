from pathlib import Path
from typing import Self

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion


class OpenAiSTTClient:
    def __init__(self):
        self.client = OpenAI()
        self.model = "whisper-1"
        self.audio_file = None

    def set_audio_file(self, file: Path) -> Self:
        self.audio_file = file
        return self

    def call(self, **kwargs) -> str:
        audio_file = open(self.audio_file, "rb")
        transcription = self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            **kwargs
        )
        return transcription.text


class OpenAiClient:
    def __init__(self):
        self.client = OpenAI()
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
        if (_parsing := kwargs.get('parsing')) is not None:
            _ = kwargs.pop('parsing')
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **kwargs
        )
        if _parsing:
            return completion.choices[0].message.content
        return completion
