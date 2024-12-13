import json
from json.decoder import JSONDecodeError
from pathlib import Path

from core.client.openai.chat import OpenAIClient
from core.prompt_manager import PromptManager
from utile.progress_bar import ProgressBar


class CaptionTextSummarizer:
    PROMPT = PromptManager()

    def __init__(self, video_file: Path, output_file: Path):
        print('Summarize Video...')
        self.output_file = output_file
        self.title = video_file.stem
        self.data = json.load(self.output_file.open('r'))
        self.open_ia = OpenAIClient()
        print(f'\tL output file : {self.output_file}')
        print(f'\tL title : {self.title}')

    def _build_prompt(self) -> str:
        prompt = ''
        prompt += "<Title>" + self.title + "</Title>" + '\n'

        for row in ProgressBar(self.data.get('caption'), bar_length=50, prefix='\t'):
            if (row.get('desc') is None) | (row.get('text') is None):
                continue
            prompt += "<VideoFrame>" + '\n'
            prompt += "\t<Caption>" + row.get('desc').replace('\n', ' ') + "</Caption>" + '\n'
            prompt += "\t<Subtitle>" + row.get('text').replace('\n', ' ') + "</Subtitle>" + '\n'
            prompt += "</VideoFrame>" + '\n'
        return prompt

    def run(self):
        user_prompt = self._build_prompt()
        self.open_ia.add_prompt(role='system', text=self.PROMPT.SUMMARY)
        self.open_ia.add_prompt(role='user', text=user_prompt)
        print('\tL request progress...')

        response = self.open_ia.call(response_format={"type": "json_object"}, temperature=1.0, parsing=True)
        try:
            response = json.loads(response)
        except JSONDecodeError as e:
            print('json parse error')
            print(response)
            raise e

        self.data.update(response)
        self.output_file.open('w').write(
            json.dumps(self.data, ensure_ascii=False, indent=2)
        )
