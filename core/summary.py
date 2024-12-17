import json
from json.decoder import JSONDecodeError
from pathlib import Path

from core.client.openai.chat import OpenAIClient
from core.prompt_manager import PromptManager
from utile.progress_bar import ProgressBar


class CaptionTextSummarizer:
    PROMPT = PromptManager()

    def __init__(self, output_file: Path):
        print('Summarize Video...')
        self.output_file = output_file
        self.data = json.load(self.output_file.open('r'))
        self.open_ia = OpenAIClient()
        print(f'\tL output file : {self.output_file}')

    def _build_prompt(self) -> str:
        prompt = ''
        prompt += "<Title>" + self.data.get('title') + "</Title>" + "\n"
        prompt += "<VideoFrames>" + "\n"
        for row in ProgressBar(self.data.get('caption'), bar_length=50, prefix='\t'):
            if (row.get('desc') is None) | (row.get('text') is None):
                continue
            _id = row.get('groupId')
            prompt += f"  <VideoFrame{_id}>" + "\n"
            prompt += "    <Caption>" + row.get('desc').replace('\n', ' ') + "</Caption>" + "\n"
            prompt += "    <Subtitle>" + row.get('text').replace('\n', ' ') + "</Subtitle>" + "\n"
            prompt += f"  </VideoFrame{_id}>" + "\n"
        prompt += "</VideoFrames>"
        prompt += "<Speech>" + self.data.get('stt') + "</Speech>" + "\n"
        return prompt

    def run(self):
        user_prompt = self._build_prompt()
        self.open_ia.add_prompt(role='system', text=self.PROMPT.SUMMARY_SYSTEM)
        self.open_ia.add_prompt(role='user', text=self.PROMPT.SUMMARY_USER)
        self.open_ia.add_prompt(role='assistant', text=self.PROMPT.SUMMARY_ASSIST)
        self.open_ia.add_prompt(role='system', text=self.PROMPT.SUMMARY_SYSTEM)
        self.open_ia.add_prompt(role='user', text=user_prompt)
        print('\tL request progress...')

        response = self.open_ia.call(
            response_format={"type": "json_object"},
            temperature=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            parsing=True)
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
