from pathlib import Path
from typing import Self

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from model.image import ImageHandler


class OpenAiVisionClient:

    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o"
        self.contents = []

    def clear(self) -> None:
        self.contents = []

    def add_prompt(self, prompt) -> Self:
        self.contents.append(
            {
                "type": "text",
                "text": prompt,
            },
        )
        return self

    def add_image(self, image: ImageHandler) -> Self:
        base64_image = image.encoding()
        self.contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
        )
        return self

    def call(self, **kwargs) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": 'user', "content": self.contents}
            ],
            **kwargs
        )


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

    def call(self, **kwargs) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **kwargs
        )
        return completion.choices[0].message.content
