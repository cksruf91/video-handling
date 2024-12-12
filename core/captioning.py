import json
import sys
import time
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any

import polars as pl

from client.openai.vision import OpenAIVisionClient, OpenAIBatchVisionClient
from core.prompt_manager import PromptManager
from model.video import Video
from utile.progress_bar import ProgressBar


class ImageCaptionWriter:
    PROMPT = PromptManager()

    def __init__(self, video_file: Path, image_dir: Path, output_file: Path):
        print("Captioning video...")
        self.output_file = output_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir = image_dir.joinpath('frames')
        self.video = Video(video_file)
        self.open_ai = OpenAIVisionClient()
        self.group_id = self._extract_group_id()

        print(f'\tL target file : {video_file}')
        print(f'\tL save dir : {self.image_dir}')
        print(f'\tL output file : {self.output_file}')

    def _extract_group_id(self) -> list[int]:
        group_ids = set()
        for _file in self.image_dir.glob('*.jpg'):
            group_ids.add(int(_file.name.split('_')[1]))
        return list(group_ids)

    def run(self):
        video_desc = []
        for gid in ProgressBar(self.group_id, bar_length=50, prefix='\t'):
            self.open_ai.prompt.clear()
            self.open_ai.prompt.add_text(self.PROMPT.CAPTIONING)

            times = []
            files = sorted(list(self.image_dir.glob(f'frame_{gid:04d}_*.jpg')))
            for file_name in files:
                times.append(file_name.name.split('_')[-2])
                self.open_ai.prompt.add_image(file_name)

            response = self.open_ai.call()  # temperature=0.3,
            try:
                content = json.loads(response.choices[0].message.content)
            except JSONDecodeError as e:
                print('json parse error, gid : {}'.format(gid))
                content = {
                    "error": str(e) + ' ' + response.choices[0].message.content,
                }
            content.update({
                'position': f"{min(times)}~{max(times)}",
                'groupId': gid,
                'text': ' '.join(content['text']) if isinstance(content['text'], list) else content['text']
            })
            video_desc.append(content)
        desc_df = pl.DataFrame(video_desc, orient='row').sort('position')
        self._save(desc_df.to_dicts(), self.output_file)

    @staticmethod
    def _save(result: list[dict[str, str]], file: Path):
        data = {
            'caption': result
        }
        json.dump(data, file.open('w'), ensure_ascii=False, indent=2)


class BatchImageCaptionWriter(ImageCaptionWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_file_dir = Path('temp')
        self._batch_file_dir.parent.mkdir(parents=True, exist_ok=True)
        self.open_ai = OpenAIBatchVisionClient(self._batch_file_dir)
        self.open_ai.flush_file()

    def create_batch_file(self) -> dict[str, dict[str, Any]]:
        requests = {}
        for gid in ProgressBar(self.group_id, bar_length=50, prefix='\t create batch file'):
            self.open_ai.prompt.add_text(self.PROMPT.CAPTIONING)

            times = []
            files = sorted(list(self.image_dir.glob(f'frame_{gid:04d}_*.jpg')))
            for file_name in files:
                times.append(file_name.name.split('_')[-2])
                self.open_ai.prompt.add_image(file_name)
            _id = f"request_{gid}"
            requests[_id] = {
                'time': f"{min(times)}~{max(times)}", 'groupId': gid
            }
            self.open_ai.write_prompt(request_id=_id)
        return requests

    def run(self):
        requests = self.create_batch_file()
        self.open_ai \
            .upload() \
            .create_batch() \
            .flush_file()

        i = 0
        while True:
            i += 1
            status = self.open_ai.get_status()
            sys.stdout.write(f'\r\tL {i}] create batch status: {status}')
            if all([s == 'completed' for b, s in status.items()]):
                print('')
                break
            if any([s == 'failed' for b, s in status.items()]):
                raise RuntimeError(f'create batch failed : {self.open_ai.batch_objects}')
            time.sleep(10)

        print(f'\tL retrieving...')
        result = []
        for line in self.open_ai.retrieve():
            try:
                content = json.loads(line['response']['body']['choices'][0]['message']['content'])
            except (JSONDecodeError, KeyError) as e:
                print('json parse error, request_id : {}'.format(line['custom_id']))
                content = {
                    "error": str(e) + ' ' + json.dumps(line, ensure_ascii=False),
                }
            content.update({'requestId': line['custom_id']})
            content.update(requests[line['custom_id']])
            result.append(content)

        self._save(result, self.output_file)

        error_file = self.output_file.parent.joinpath(self.output_file.stem + '_error.jsonl')
        error = [e for e in self.open_ai.retrieve_error()]
        if error:
            self._save(result, error_file)

        self.open_ai.delete_files()
