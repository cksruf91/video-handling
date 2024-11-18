import argparse
from pathlib import Path

from chunking import VideoChunker
from describe import ImageDescriptor
from extract_audio import AudioTextExtractor


class Arguments(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('-t', '--task', default='chunk', choices=['chunk', 'desc', 'stt'])
        self.add_argument('-v', '--video', type=str)
        self.add_argument('-s', '--save', type=str)
        self.add_argument('-i', '--image', type=str, default=None)
        self.add_argument('-a', '--audio', type=str, default=None)
        arg = self.parse_args()

        self.task = arg.task
        self.video = Path(arg.video)
        self.save = Path(arg.save)
        if arg.image is not None:
            self.image = Path(arg.image)
        if arg.audio is not None:
            self.audio = Path(arg.audio)


class Main:
    def __init__(self):
        self.args = Arguments()

    def run(self):
        self.args.save.parent.mkdir(exist_ok=True, parents=True)
        if self.args.task == 'chunk':
            task = VideoChunker(
                video=self.args.video, save_dir=self.args.save
            )
        elif self.args.task == 'desc':
            task = ImageDescriptor(
                image_dir=self.args.image, video_file=self.args.video, save_file=self.args.save
            )
        elif self.args.task == 'stt':
            self.args.audio.parent.mkdir(exist_ok=True, parents=True)
            task = AudioTextExtractor(
                file=self.args.video, audio_file=self.args.audio, save_file=self.args.save
            )
        else:
            raise ValueError('Task must be either "chunk" or "desc" or "stt"')

        task.run()


if __name__ == '__main__':
    Main().run()
