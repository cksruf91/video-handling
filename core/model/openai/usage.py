from typing import Self
from dataclasses import dataclass, field, fields

from core.model.openai.price import Pricing


@dataclass
class Usage:
    prompt_tokens: int = field(default=0)
    cached_prompt_tokens: int = field(default=0)
    completion_tokens: int = field(default=0)
    image_count: int = field(default=0)
    audio_length: int = field(default=0)
    model: str = field(default='', init=False)

    _1m: int = field(default=1000000, init=False, repr=False)
    _amount: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def prompt_amount(self, price: Pricing) -> float:
        return price.prompt_price * (self.prompt_tokens / self._1m)

    def cached_prompt_amount(self, price: Pricing) -> float:
        return price.cached_prompt_price * (self.cached_prompt_tokens / self._1m)

    def completion_amount(self, price: Pricing) -> float:
        return price.completion_price * (self.completion_tokens / self._1m)

    def image_amount(self, price: Pricing) -> float:
        return price.image_1080_1920_price * self.image_count

    def audio_amount(self, price: Pricing) -> float:
        return price.audio_price * (self.audio_length / 60)

    def pricing(self, price: Pricing) -> Self:
        self.prompt_tokens -= self.cached_prompt_tokens
        self._amount = {
            'prompt': self.prompt_amount(price),
            'cached': self.cached_prompt_amount(price),
            'completion': self.completion_amount(price),
            'image': self.image_amount(price),
            'audio': self.audio_amount(price),
        }
        self.model = price.name
        return self

    def total(self) -> float:
        return self._amount.get('prompt') + \
            self._amount.get('cached') + \
            self._amount.get('completion') + \
            self._amount.get('image') + \
            self._amount.get('audio')

    def usages(self):
        usage = {}
        for f in fields(self):
            if f.repr and (getattr(self, f.name) != 0):
                if (f.type == int) or (f.type == float):
                    usage[f.name] = f"{getattr(self, f.name):,}"
                else:
                    usage[f.name] = f"{getattr(self, f.name)}"
        return usage
