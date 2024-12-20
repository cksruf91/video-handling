import json
from pathlib import Path

from core.model.openai.price import GPT4oPrice, GPT4oMiniPrice, WhisperPrice
from core.model.openai.usage import Usage


class Price:
    def __init__(self, output_file: Path):
        print('Calculating price...')
        self.data = json.load(output_file.open('r'))
        print(f'\tL output file : {output_file}')

    def run(self) -> None:
        captioning = Usage()
        summary = Usage()
        audio = Usage()

        for caption in self.data['caption']:
            captioning.prompt_tokens += caption['cost']['prompt_tokens']
            captioning.completion_tokens += caption['cost']['completion_tokens']
            captioning.image_count += caption['cost']['image']

        summary.prompt_tokens = self.data['summary_cost']['prompt_tokens']
        summary.completion_tokens = self.data['summary_cost']['completion_tokens']
        audio.audio_length += self.data['audio_length'] // 60

        captioning.pricing(GPT4oPrice())
        summary.pricing(GPT4oMiniPrice())
        audio.pricing(WhisperPrice())

        print("\tL total: {:3.6f}$".format(captioning.total() + summary.total() + audio.total()))
        print("\t L captioning: {:3.6f}$".format(captioning.total()))
        print("\t   L tokens: {}".format(captioning.to_json()))
        print("\t L summary: {:3.6f}$".format(summary.total()))
        print("\t   L tokens: {}".format(summary.to_json()))
        print("\t L audio: {:3.6f}$".format(audio.total()))
        print("\t   L audio: {}".format(audio.to_json()))
