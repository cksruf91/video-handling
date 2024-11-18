import argparse
from pathlib import Path

from core.chunking import VideoChunker
from core.describe import ImageDescriptor
from core.extract_audio import AudioTextExtractor
from core.extract_keyword import KeywordExtractor


class Arguments(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('-t', '--task', default='chunk', choices=['chunk', 'desc', 'stt', 'keyword'])
        self.add_argument('-v', '--video', type=str, default=None, help='Video file to read')
        self.add_argument('-o', '--output', type=str, default=None, help='output parquet file')
        self.add_argument('-s', '--stt', type=str, default=None, help='stt result text file')
        self.add_argument('-i', '--image', type=str, default=None, help='video chunking result store directory')
        self.add_argument('-a', '--audio', type=str, default=None, help='extracted audio file')
        arg = self.parse_args()

        self.task = arg.task

        if arg.video is not None:
            self.video = Path(arg.video)
        if arg.stt is not None:
            self.stt = Path(arg.stt)
        if arg.output is not None:
            self.output = Path(arg.output)
        if arg.image is not None:
            self.image = Path(arg.image)
        if arg.audio is not None:
            self.audio = Path(arg.audio)


class Main:
    def __init__(self):
        self.args = Arguments()

    def run(self):
        if self.args.task == 'chunk':
            self.args.image.mkdir(exist_ok=True, parents=True)
            task = VideoChunker(
                video_file=self.args.video, save_dir=self.args.image
            )
        elif self.args.task == 'desc':
            self.args.output.parent.mkdir(exist_ok=True, parents=True)
            task = ImageDescriptor(
                image_dir=self.args.image, video_file=self.args.video, save_file=self.args.output
            )
        elif self.args.task == 'keyword':
            task = KeywordExtractor(output_file=self.args.output)
        elif self.args.task == 'stt':
            self.args.stt.parent.mkdir(exist_ok=True, parents=True)
            self.args.audio.parent.mkdir(exist_ok=True, parents=True)
            task = AudioTextExtractor(
                file=self.args.video, audio_file=self.args.audio, save_file=self.args.stt
            )
        else:
            raise ValueError('Task must be either "chunk","desc","keyword" or "stt"')

        task.run()


if __name__ == '__main__':
    Main().run()
