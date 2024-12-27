import base64
import json
import os
from pathlib import Path
from typing import Self, Iterable, Any

from openai import OpenAI
from openai.types import Batch, FileObject
from openai.types.chat.chat_completion import ChatCompletion

from core.model.image import ImageHandler


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
        self.client = OpenAI(api_key=os.environ.get("OPENAI_KEY_VIDEO"))
        self.prompt = _Prompt()
        self.model = "gpt-4o-mini"

    def call(self, **kwargs) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": 'user', "content": self.prompt.contents}
            ],
            **kwargs
        )


class OpenAIBatchVisionClient:
    MAX_REQUESTS = 50000
    MAX_FILE_SIZE = 190 * 1024 * 1024  # 200 MB

    def __init__(self, batch_file_dir: Path):
        super().__init__()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_KEY_VIDEO"))
        self.prompt = _Prompt()
        self.model = "gpt-4o"
        self._no = 0
        self.batch_file_dir = batch_file_dir
        self.file_objects: list[FileObject] = []
        self.batch_objects: list[Batch] = []

    def flush_file(self) -> Self:
        for file in self.batch_file_dir.glob('*.jsonl'):
            file.unlink(missing_ok=True)
        return self

    @property
    def batch_file(self) -> Path:
        return self.batch_file_dir.joinpath(f'batch_file_{self._no}.jsonl')

    def write_request(self, request_id: str, **kwargs) -> Self:
        _body = {
            "model": self.model,
            "messages": [{"role": 'user', "content": self.prompt.contents}],
            "max_tokens": 3000,
        }
        _body.update(kwargs)
        line = json.dumps({
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": _body
        }, ensure_ascii=False)
        self.batch_file.open('a').write(line + '\n')
        self.prompt.clear()
        if self.batch_file.stat().st_size >= self.MAX_FILE_SIZE:
            self._no += 1
        return self

    def upload(self) -> Self:
        for _file in self.batch_file_dir.glob('*.jsonl'):
            print(f"\t upload file: {_file}")
            self.file_objects.append(
                self.client.files.create(
                    file=_file.open('rb'), purpose='batch'
                )
            )
        return self

    def create_batch(self) -> Self:
        for file in self.file_objects:
            print(f"\t create batch: {file.filename}")
            self.batch_objects.append(
                self.client.batches.create(
                    input_file_id=file.id,
                    endpoint='/v1/chat/completions',
                    completion_window='24h',
                    metadata={"description": "image captioning job"}
                )
            )
        return self

    def get_status(self) -> dict[str, str]:
        for i in range(len(self.batch_objects)):
            self.batch_objects[i] = self.client.batches.retrieve(self.batch_objects[i].id)
        return {batch.id: batch.status for batch in self.batch_objects}

    def retrieve(self) -> Iterable[dict[str, Any]]:
        for batch in self.batch_objects:
            file_response = self.client.files.content(batch.output_file_id)
            for line in file_response.iter_lines():
                yield json.loads(line)

    def retrieve_error(self) -> Iterable[dict[str, Any]]:
        for batch in self.batch_objects:
            if batch.error_file_id is not None:
                file_response = self.client.files.content(batch.error_file_id)
                for line in file_response.iter_lines():
                    yield json.loads(line)

    def delete_files(self) -> Self:
        for file in self.file_objects:
            output = self.client.files.delete(file.id)
            print(f"delete result: {output}")
        for file in self.batch_objects:
            output = self.client.files.delete(file.output_file_id)
            print(f"delete result: {output}")
        return self
