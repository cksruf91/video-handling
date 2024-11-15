import argparse
from pathlib import Path

from chunking import VideoChunker


class Arguments(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument(
            '-b', '--batch', default='split', choices=['split', 'prompt']
        )
        self.add_argument('-f', '--file', type=str)
        self.add_argument('-s', '--save', type=str)
        arg = self.parse_args()

        self.batch = arg.batch
        self.file = Path(arg.file)
        self.save = Path(arg.save)


class Main:
    def __init__(self):
        self.args = Arguments()

    def run(self):
        if self.args.batch == 'split':
            self.args.save.mkdir(exist_ok=True, parents=True)
            batch = VideoChunker(self.args.file, self.args.save)
        else:
            raise ValueError('Batch must be either "split" or "prompt"')

        batch.run()


if __name__ == '__main__':
    Main().run()
