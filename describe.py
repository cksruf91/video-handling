import json
from pathlib import Path

import polars as pl

from model.image import ImageHandler
from model.openai_client import OpenAiVisionClient
from model.video import Video
from utile.progress_bar import ProgressBar


class ImageDescriptor:
    def __init__(self, image_dir: Path, video_file: Path, save_file: Path):
        self.save_file = save_file
        self.save_file.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir = image_dir
        self.video = Video(video_file)
        self.open_ai = OpenAiVisionClient()
        self.group_id = self._extract_group_id()

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
        return list(group_ids)[:10]

    def to_time_string(self, frame_nums: tuple[int, int]) -> str:
        def minutes_sec(s):
            sec = round(s / self.video.fps)
            return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

        start = frame_nums[0]
        end = frame_nums[1]

        return f"{minutes_sec(start)} ~ {minutes_sec(end)}"

    def run(self):
        video_desc = []
        for gid in ProgressBar.init(self.group_id):
            self.open_ai.clear()
            self.open_ai.add_prompt(self.prompt)

            frame_no = []
            files = sorted(list(self.image_dir.glob(f'frame_{gid:04d}_*.jpg')))
            for file_name in files:
                frame_no.append(int(file_name.name.split('_')[-1].replace('.jpg', '')))
                frame = ImageHandler(file_name).rgb()
                self.open_ai.add_image(frame)

            response = self.open_ai.call()
            try:
                answer = json.loads(response.choices[0].message.content)
            except json.decoder.JSONDecodeError as e:
                answer = {
                    "desc": response.choices[0].message.content,
                    "text": str(e)
                }
            answer['frame'] = (min(frame_no), max(frame_no))
            answer['groupId'] = gid
            video_desc.append(answer)

        for vd in video_desc:
            if isinstance(vd['text'], list):
                vd['text'] = ' '.join(vd['text'])

        desc_df = pl.DataFrame(
            video_desc, orient='row'
        ).with_columns(
            pl.col('frame').map_elements(self.to_time_string, return_dtype=pl.String).alias('time')
        ).sort('time')

        desc_df.write_parquet(self.save_file)
