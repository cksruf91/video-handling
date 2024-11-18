import argparse
from pathlib import Path

from chunking import VideoChunker
from describe import ImageDescriptor


class Arguments(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('-b', '--batch', default='chunk', choices=['chunk', 'desc'])
        self.add_argument('-v', '--video', type=str)
        self.add_argument('-s', '--save', type=str)
        self.add_argument('-i', '--image', type=str, default=None)
        arg = self.parse_args()

        self.batch = arg.batch
        self.video = Path(arg.video)
        self.save = Path(arg.save)
        if arg.image is not None:
            self.image = Path(arg.image)


class Main:
    def __init__(self):
        self.args = Arguments()

    def run(self):
        if self.args.batch == 'chunk':
            self.args.save.mkdir(exist_ok=True, parents=True)
            batch = VideoChunker(
                video=self.args.video, save_dir=self.args.save
            )
        elif self.args.batch == 'desc':
            batch = ImageDescriptor(
                image_dir=self.args.image, video_file=self.args.video, save_file=self.args.save
            )
        else:
            raise ValueError('Batch must be either "chunk" or "desc"')

        batch.run()


if __name__ == '__main__':
    Main().run()
