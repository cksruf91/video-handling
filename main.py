import argparse
from pathlib import Path

from core.chunking import VideoChunker
from core.captioning import BatchImageCaptionWriter
from core.extract_audio import AudioTextExtractor
from core.extract_keyword import KeywordExtractor


class Arguments(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('-t', '--task', type=str, nargs='+', default='chunk',
                          choices=['chunk', 'cap', 'stt', 'keyword', 'all'])
        self.add_argument('-i', '--input', type=str, help='Video file to read')
        self.add_argument('-o', '--output', type=str, help='output json file')
        self.add_argument('-p', '--temp', type=str, help='temporary directory for jpg, mp3 files')
        arg = self.parse_args()

        self.task = arg.task
        self.input = Path(arg.input)
        self.output = Path(arg.output)
        self.temp = Path(arg.temp)


class Main:
    def __init__(self):
        self.args = Arguments()

    def run(self):
        print(f"task: {self.args.task}")
        if ('chunk' in self.args.task) or ('all' in self.args.task):
            VideoChunker(
                video_file=self.args.input, image_dir=self.args.temp
            ).run()
        if ('cap' in self.args.task) or ('all' in self.args.task):
            BatchImageCaptionWriter(
                video_file=self.args.input, image_dir=self.args.temp, output_file=self.args.output
            ).run()
        if ('keyword' in self.args.task) or ('all' in self.args.task):
            KeywordExtractor(output_file=self.args.output).run()
        if ('stt' in self.args.task) or ('all' in self.args.task):
            AudioTextExtractor(
                video_file=self.args.input, audio_dir=self.args.temp, output_file=self.args.output
            ).run()


if __name__ == '__main__':
    Main().run()
