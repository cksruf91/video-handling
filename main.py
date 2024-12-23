import argparse
from pathlib import Path

from core.captioning import ImageCaptionWriter, BatchImageCaptionWriter
from core.chunking import VideoChunker
from core.extract_audio import AudioTextExtractor
from core.extract_keyword import KeywordExtractor
from core.pricing import Price
from core.summary import Summarizer


class Arguments(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()
        self.add_argument('-t', '--task', type=str, nargs='+', default='chunk',
                          choices=['chunk', 'cap', 'stt', 'keyword', 'summary', 'all', 'price'])
        self.add_argument('-i', '--input', type=str, help='Video file to read')
        self.add_argument('-o', '--output', type=str, help='output json file')
        self.add_argument('-p', '--temp', type=str, help='temporary directory for jpg, mp3 files')
        self.add_argument('-b', '--batch_api', action='store_true',
                          help='use batch api')
        arg = self.parse_args()

        self.task = arg.task
        self.input = Path(arg.input)
        self.output = Path(arg.output)
        self.temp = Path(arg.temp)
        self.batch_api = arg.batch_api


class Main:
    def __init__(self):
        self.args = Arguments()

    def run(self):
        if 'all' in self.args.task:
            self.args.task.extend(['chunk', 'cap', 'stt', 'summary'])
        print(f"task: {self.args.task}")
        if 'chunk' in self.args.task:
            VideoChunker(
                video_file=self.args.input, image_dir=self.args.temp
            ).run()
        if 'cap' in self.args.task:
            task = BatchImageCaptionWriter if self.args.batch_api else ImageCaptionWriter
            task(
                video_file=self.args.input, image_dir=self.args.temp, output_file=self.args.output
            ).run()
        if 'keyword' in self.args.task:
            KeywordExtractor(output_file=self.args.output).run()
        if 'stt' in self.args.task:
            AudioTextExtractor(
                video_file=self.args.input, audio_dir=self.args.temp, output_file=self.args.output
            ).run()
        if 'summary' in self.args.task:
            Summarizer(output_file=self.args.output).run()
        if 'price' in self.args.task:
            Price(output_file=self.args.output).run()


if __name__ == '__main__':
    Main().run()
