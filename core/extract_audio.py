import json
from pathlib import Path

from client.openai.stt import OpenAISTTClient
from model.audio import Audio


class AudioTextExtractor:
    def __init__(self, video_file: Path, audio_dir: Path, output_file: Path):
        print('Extracting text from audio...')
        self.open_ai = OpenAISTTClient()
        self.audio_handler = Audio(video_file)
        self.audio_file = audio_dir.joinpath('audio.mp3')
        self.audio_file.parent.mkdir(exist_ok=True, parents=True)
        self.output_file = output_file
        self.output = json.load(self.output_file.open('r'))
        print(f'\tL audio file : {self.audio_file}')
        print(f'\tL output file : {self.output_file}')

    def run(self) -> None:
        print('\tL extracting audio files...')
        self.audio_handler.extract_audio(self.audio_file)
        print('\tL STT request progress...')
        text = self.open_ai \
            .set_audio_file(self.audio_file) \
            .call(temperature=0)
        self.output.append({'stt': text})
        print('\tL Done')
        json.dump(self.output, self.output_file.open('w'), ensure_ascii=False, indent=2)
