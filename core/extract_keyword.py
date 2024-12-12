import json
from pathlib import Path

from client.openai.chat import OpenAIClient
from core.prompt_manager import PromptManager
from utile.progress_bar import ProgressBar


class KeywordExtractor:
    _PROMPT = PromptManager()

    def __init__(self, output_file: Path):
        print('Extracting keywords...')
        self.output_file = output_file
        self.data = json.load(self.output_file.open('r'))
        self.open_ai = OpenAIClient()
        print(f'\tL output file : {self.output_file}')

    def run(self) -> None:
        caption = self.data.get('caption')
        for row in ProgressBar(caption, max_value=len(caption), bar_length=50, prefix='\t'):
            if (row.get('desc') is None) | (row.get('text') is None):
                continue
            full_text = row['desc'] + ' ' + row['text']
            self.open_ai \
                .add_prompt(role='system', text=self._PROMPT.KEYWORD_SYSTEM) \
                .add_prompt(role='user', text=self._PROMPT.KEYWORD_USER.format(full_text=full_text))
            row['keyword'] = self.open_ai.call()
        json.dump(self.data, self.output_file.open('w'), ensure_ascii=False, indent=2)
