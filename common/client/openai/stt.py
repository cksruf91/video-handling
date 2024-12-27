import os
from pathlib import Path
from typing import Self

from openai import OpenAI
from openai.types.audio import Transcription


class OpenAISTTClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_KEY_VIDEO"))
        self.model = "whisper-1"
        self.audio_file = None

    def set_audio_file(self, file: Path) -> Self:
        self.audio_file = file
        return self

    def call(self, **kwargs) -> Transcription | str:
        _parsing = kwargs.pop('parsing') if 'parsing' in kwargs else True
        audio_file = open(self.audio_file, "rb")
        transcription = self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            **kwargs
        )
        if _parsing:
            return transcription.text
        return transcription
