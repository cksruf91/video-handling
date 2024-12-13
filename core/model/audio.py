from pathlib import Path

from moviepy.editor import VideoFileClip
from pydub import AudioSegment


class AudioMoviepy:
    """ deprecated """

    def __init__(self, file: Path):
        self.video = VideoFileClip(str(file))

    def extract_audio(self, audio_path: Path):
        self.video.audio.write_audiofile(str(audio_path))


class Audio:
    def __init__(self, file: Path, file_format: str = 'mp4'):
        self.audio = AudioSegment.from_file(file, file_format)

    def extract_audio(self, audio_file: Path, file_format: str = 'mp3'):
        self.audio.export(audio_file, format=file_format)
