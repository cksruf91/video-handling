from pathlib import Path

import polars as pl

from model.openai_client import OpenAiClient
from utile.progress_bar import ProgressBar


class KeywordExtractor:
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.frame = pl.read_parquet(self.output_file)
        self.open_ai = OpenAiClient()

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
        keywords = []
        for row in ProgressBar(self.frame.iter_rows(named=True), max_value=len(self.frame), bar_length=50):
            full_text = row['desc'] + row['text']
            self.open_ai \
                .add_prompt(role='system', text=self.system_prompt) \
                .add_prompt(role='system', text=self.user_prompt.format(full_text=full_text))
            keywords.append(self.open_ai.call())

        self.frame.with_columns(
            pl.Series(name='keyword', values=keywords)
        ).write_parquet(self.output_file)
