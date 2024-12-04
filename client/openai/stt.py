from pathlib import Path
from typing import Self

from openai import OpenAI


class OpenAISTTClient:
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
