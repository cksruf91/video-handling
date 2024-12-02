import base64
import json
from pathlib import Path
from typing import Self, Iterable, Any

from openai import OpenAI
from openai.types import Batch, FileObject
from openai.types.chat.chat_completion import ChatCompletion

from model.image import ImageHandler


class _Prompt:

    def __init__(self):
        self.contents = []

    def add_text(self, prompt) -> Self:
        self.contents.append(
            {
                "type": "text",
                "text": prompt,
            },
        )
        return self

    def add_image(self, image: ImageHandler | Path) -> Self:
        if isinstance(image, ImageHandler):
            base64_image = image.encoding()
        elif isinstance(image, Path):
            base64_image = base64.b64encode(image.open('rb').read()).decode('utf-8')
        else:
            raise RuntimeError('image must be ImageHandler or Path')

        self.contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
        )
        return self

    def clear(self) -> Self:
        self.contents = []
        return self


class OpenAIVisionClient:
    def __init__(self):
        super().__init__()
        self.client = OpenAI()
        self.prompt = _Prompt()
        self.model = "gpt-4o"

    def call(self, **kwargs) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": 'user', "content": self.prompt.contents}
            ],
            **kwargs
        )


class OpenAIBatchVisionClient:
    limitation = 50000

    def __init__(self, batch_file: Path):
        super().__init__()
        self.client = OpenAI()
        self.prompt = _Prompt()
        self.model = "gpt-4o"
        self.batch_file = batch_file
        self._id = 0
        self.upload_response: FileObject | None = None
        self.batch_create: Batch | None = None

    def flush_file(self) -> Self:
        self.batch_file.unlink(missing_ok=True)
        return self

    def write_prompt(self, request_id: str) -> Self:
        line = json.dumps({
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [{"role": 'user', "content": self.prompt.contents}],
                "max_tokens": 1000
            }
        }, ensure_ascii=False)
        self.batch_file.open('a').write(line + '\n')
        self.prompt.clear()
        return self

    def upload(self) -> Self:
        self.upload_response = self.client.files.create(
            file=self.batch_file.open('rb'), purpose='batch'
        )
        return self

    def create_batch(self) -> Self:
        self.batch_create = self.client.batches.create(
            input_file_id=self.upload_response.id,
            endpoint='/v1/chat/completions',
            completion_window='24h',
            metadata={
                "description": "image captioning job"
            }
        )
        return self

    def checking(self) -> str:
        self.batch_create = self.client.batches.retrieve(self.batch_create.id)
        return self.batch_create.status

    def retrieve(self) -> Iterable[dict[str, Any]]:
        file_response = self.client.files.content(self.batch_create.output_file_id)
        for line in file_response.iter_lines():
            yield json.loads(line)
