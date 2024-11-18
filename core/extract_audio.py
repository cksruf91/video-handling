from pathlib import Path

from model.audio import Audio
from model.openai_client import OpenAiSTTClient


class AudioTextExtractor:
    def __init__(self, file: Path, audio_file: Path, save_file: Path):
        self.open_ai = OpenAiSTTClient()
        self.audio_handler = Audio(file)
        self.audio_file = audio_file
        self.save_file = save_file

    def run(self) -> None:
        print(f'Extracting audio -> {self.audio_file}')
        self.audio_handler.extract_audio(self.audio_file)
        print('\tL Done')
        print(f'STT request progress...')
        text = self.open_ai \
            .set_audio_file(self.audio_file) \
            .call()
        self.write_text(text=text)
        print(f'\tL output -> {self.save_file}')

    def write_text(self, text: str) -> None:
        with open(self.save_file, 'w') as f:
            f.write(text)
