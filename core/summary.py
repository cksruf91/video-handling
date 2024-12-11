import json
from json.decoder import JSONDecodeError
from pathlib import Path

from client.openai.chat import OpenAIClient
from utile.progress_bar import ProgressBar

SUMMARY_PROMPT = """
제공된 동영상에서 추출된 **동영상의 프레임**들의 내용들을 가지고 동영상의 전체적인 내용의 요약 및 해시태그를 생성해주세요.

##지침
1. **동영상 프레임** 은 **Caption** 과 *Subtitle* 로 구성 되어 있습니다.
    1-1. **Caption** 은 동영상 프레임에 대한 전체적인 설명을 의미 합니다.
    1-2 **Subtitle** 은 해당 프레임에 있는 자막의 내용입니다.
2. **동영상 프레임**들의 **Caption**과 **Subtitle**의 내용을 확인합니다. 프레임들은 동영상의 재생순서대로 제공됩니다.
3. **동영상의 제목**을 참고하세요.
4. **동영상 프레임**들의 설명 내용들을 가지고 동영상의 **전체적인 내용의 요약**을해주세요.
5. 동영상의 전체적인 내용을 가지고 **해시태그**를 약 20개 생성해주세요.
6. 응답값은 아래의 **JSON** 포멧을 따라 주세요.
```
{
    "summary": "동영상에 대한 요약글",
    "hashtag": ["해시태그1", "해시태그2", ..... ]
}
```

##참고
- **해시태그**는 SNS(소셜네트워크서비스) 에서 사람들에게 해당 영상을 홍보하기 위한 목적이니 참고해주세요. 
- <title>은 **동영상의 제목** 을 의미합니다.
- <video frame>은 **동영상 프레임**을 의미합니다.

** 동영상 프레임 **:
"""


class CaptionTextSummarizer:

    def __init__(self, video_file: Path, output_file: Path):
        print('Summarize Video...')
        self.output_file = output_file
        self.title = video_file.stem
        self.data = json.load(self.output_file.open('r'))
        self.open_ia = OpenAIClient()
        print(f'\tL output file : {self.output_file}')
        print(f'\tL title : {self.title}')

    def _build_prompt(self) -> str:
        prompt = ''
        prompt += "</title>" + self.title + "<title>" + '\n'

        for row in ProgressBar(self.data, bar_length=50, prefix='\t'):
            if (row.get('desc') is None) | (row.get('text') is None):
                continue
            prompt += "</video frame>" + '\n'
            prompt += "\t</Caption>" + row.get('desc').replace('\n', ' ') + "<Caption>" + '\n'
            prompt += "\t</Subtitle>" + row.get('text').replace('\n', ' ') + "<Subtitle>" + '\n'
            prompt += "<video frame>" + '\n'
        return prompt

    def run(self):
        self.open_ia.add_prompt(role='system', text=SUMMARY_PROMPT)
        self.open_ia.add_prompt(role='user', text=self._build_prompt())
        print('\tL request progress...')
        response = self.open_ia.call(response_format={"type": "json_object"}, temperature=1.0, parsing=True)
        try:
            response = json.loads(response)
        except JSONDecodeError as e:
            print('json parse error')
            print(response)
            raise e

        self.data.append(response)
        self.output_file.open('w').write(
            json.dumps(self.data, ensure_ascii=False, indent=2)
        )
