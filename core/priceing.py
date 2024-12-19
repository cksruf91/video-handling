import json
from pathlib import Path

from core.model.openai.price import GPT4oPrice, GPT4oMiniPrice, WhisperPrice


class Price:
    def __init__(self, output_file: Path):
        print('Calculating price...')
        self.data = json.load(output_file.open('r'))
        self.gpt4o = GPT4oPrice()
        self.gpt4o_mini = GPT4oMiniPrice()
        self.whisper = WhisperPrice()
        print(f'\tL output file : {output_file}')

    def run(self) -> None:
        print('Price calculated')
        for caption in self.data['caption']:
            self.gpt4o.prompt_tokens += caption['cost']['prompt_tokens']
            self.gpt4o.completion_tokens += caption['cost']['completion_tokens']
            self.gpt4o.image_count += caption['cost']['image']

        self.gpt4o_mini.prompt_tokens = self.data['summary_cost']['prompt_tokens']
        self.gpt4o_mini.completion_tokens = self.data['summary_cost']['completion_tokens']
        self.whisper.audio_length += self.data['audio_length'] // 60

        print(f"total: {self.gpt4o.get_total() + self.whisper.get_total() + self.gpt4o_mini.get_total():3.4f} 달러")
        print(f"\tL {self.gpt4o}")
        print(f"\tL {self.gpt4o_mini}")
        print(f"\tL {self.whisper}")
