import json
from pathlib import Path

from client.openai.chat import OpenAIClient
from utile.progress_bar import ProgressBar


class KeywordExtractor:
    def __init__(self, output_file: Path):
        print('Extracting keywords...')
        self.output_file = output_file
        self.data = json.load(self.output_file.open('r'))
        self.open_ai = OpenAIClient()
        print(f'\tL output file : {self.output_file}')

        self.system_prompt = """
        당신은 언어 전문가 입니다.
        """
        self.user_prompt = """
        아래의 사항을 고려하여 중심이 되는 keyword를 추출해줘
        - keyword 는 명사를 위주로 추출
        - 콤마(",") 구분자로 하여 리스트 형태로 추출 ex) "keyword1,keyword2,keyword3..."

        {full_text}
        """

    def run(self) -> None:
        for row in ProgressBar(self.data, max_value=len(self.data), bar_length=50, prefix='\t'):
            if (row.get('desc') is None) | (row.get('text') is None):
                continue
            full_text = row['desc'] + ' ' + row['text']
            self.open_ai \
                .add_prompt(role='system', text=self.system_prompt) \
                .add_prompt(role='system', text=self.user_prompt.format(full_text=full_text))
            row['keyword'] = self.open_ai.call()
        json.dump(self.data, self.output_file.open('w'), ensure_ascii=False, indent=2)
