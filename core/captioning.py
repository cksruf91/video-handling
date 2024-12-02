import json
import time
from json.decoder import JSONDecodeError
from pathlib import Path

import polars as pl

from client.openai.vision import OpenAIVisionClient, OpenAIBatchVisionClient
from model.video import Video
from utile.progress_bar import ProgressBar


class ImageCaptionWriter:

    def __init__(self, video_file: Path, image_dir: Path, output_file: Path):
        print("Describe frames...")
        self.output_file = output_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir = image_dir.joinpath('frames')
        self.video = Video(video_file)
        self.open_ai = OpenAIVisionClient()
        self.group_id = self._extract_group_id()

        print(f'\tL target file : {video_file}')
        print(f'\tL save dir : {self.image_dir}')
        print(f'\tL output file : {self.output_file}')

        self.prompt = """
        1. 다음에 주의해서 이미지를 설명해줘
         - 이미지에 대한 설명을 할때는 각각의 이미지에 대해 따로 설명하지 말고 한번에 할 것,
         - 주어지는 이미지는 모두 영상내에서 이어지는 장면이기 때문에 하나의 이미지 인것 처럼 설명 할 것
         - 이미지에 보이는 장소가 어디인지 정확하지 않다면 장소 이름에 대한 언급을 하지 말 것 
         - 설명은 한국어로 할 것
        2. 다음에 유의해서 이미지에 포함되어 있는 자막을 text 로 추출해줘 
         - 만약 자막으로 보이는 것이 없다면 빈값으로 남겨 둘 것 
         - 자막에서 "하나투어"는 생략 할 것
         - 자막은 가능하면 있는 그대로 주며 어떠한 변환도 하지 않을 것
        3. 응답은 아래와 같은 포멧으로 줘
         - {
          "desc" : 1번에 해당하는 이미지에 대한 설명
          "text" : 2번에 해당하는 자막 텍스트
         }
        4. 응답은 반듯이 json 변환 가능해야 하며 다른 부가적인 내용은 없어야해, 코드를 JSON markers 로 감싸지마 
        """

    def _extract_group_id(self) -> list[int]:
        group_ids = set()
        for _file in self.image_dir.glob('*.jpg'):
            group_ids.add(int(_file.name.split('_')[1]))
        return list(group_ids)

    def run(self):
        video_desc = []
        for gid in ProgressBar(self.group_id, bar_length=50, prefix='\t'):
            self.open_ai.prompt.clear()
            self.open_ai.prompt.add_text(self.prompt)

            times = []
            files = sorted(list(self.image_dir.glob(f'frame_{gid:04d}_*.jpg')))
            for file_name in files:
                times.append(file_name.name.split('_')[-2])
                self.open_ai.prompt.add_image(file_name)

            response = self.open_ai.call()  # temperature=0.3,
            try:
                content = json.loads(response.choices[0].message.content)
            except JSONDecodeError as e:
                content = {
                    "desc": response.choices[0].message.content,
                    "text": str(e)
                }
            content['position'] = f"{min(times)}~{max(times)}"
            content['groupId'] = gid
            if isinstance(content['text'], list):
                content['text'] = ' '.join(content['text'])
            video_desc.append(content)

        desc_df = pl.DataFrame(video_desc, orient='row').sort('position')

        if self.output_file.suffix == '.parquet':
            desc_df.write_parquet(self.output_file)
        elif self.output_file.suffix == '.json':
            json.dump(desc_df.to_dicts(), self.output_file.open('w'), ensure_ascii=False, indent=2)
        else:
            RuntimeError(f'file extension not supported : {self.output_file.suffix}')


class BatchImageCaptionWriter(ImageCaptionWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.open_ai = OpenAIBatchVisionClient(Path('temp/batch_data.jsonl'))
        self.open_ai.flush_file()

    def run(self):
        temp = {}
        for gid in ProgressBar(self.group_id, bar_length=50, prefix='\t create batch file'):
            self.open_ai.prompt.add_text(self.prompt)

            times = []
            files = sorted(list(self.image_dir.glob(f'frame_{gid:04d}_*.jpg')))
            for file_name in files:
                times.append(file_name.name.split('_')[-2])
                self.open_ai.prompt.add_image(file_name)
            _id = f"request_{gid}"
            temp[_id] = {
                'time': f"{min(times)}~{max(times)}", 'groupId': gid
            }
            self.open_ai.write_prompt(request_id=_id)

        self.open_ai \
            .upload() \
            .create_batch() \
            .flush_file()

        while True:
            status = self.open_ai.checking()
            print(f'\tL create batch status: {status}')
            if status == 'completed':
                break
            if status == 'failed':
                raise RuntimeError(f'create batch failed : {self.open_ai.batch_create}')
            time.sleep(5)

        print(f'\tL retrieving...')
        result = []
        for line in self.open_ai.retrieve():
            try:
                content = json.loads(line['response']['body']['choices'][0]['message']['content'])
            except (JSONDecodeError, KeyError) as e:
                content = {
                    "desc": str(e) + ' ' + json.dumps(line, ensure_ascii=False),
                    "text": ''
                }
            request_id = line['custom_id']
            content['requestId'] = request_id
            content.update(temp[request_id])
            result.append(content)

        json.dump(result, self.output_file.open('w'), ensure_ascii=False, indent=2)
