from openai import OpenAI

from model.image import ImageHandler


class OpenAiVisionClient:

    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o"
        self.contents = []

    def clear(self):
        self.contents = []

    def add_prompt(self, prompt):
        self.contents.append(
            {
                "type": "text",
                "text": prompt,
            },
        )
        return self

    def add_image(self, image: ImageHandler):
        base64_image = image.encoding()
        self.contents.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            },
        )
        return self

    def call(self):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": 'user', "content": self.contents}
            ]
        )
