from dataclasses import dataclass, field


@dataclass
class _Pricing:
    input: float = field(default=0.0)
    output: float = field(default=0.0)
    image: float = field(default=0.0)
    total: float = field(default=0.0)

    prompt_price: float = field(default=None, init=False)
    prompt_tokens: int = field(default=0, init=False)
    completion_price: float = field(default=None, init=False)
    completion_tokens: int = field(default=0, init=False)
    image_1080_1920_price: float = field(default=None, init=False)
    image_count: int = field(default=0, init=False)
    _1m: int = field(default=1000000, init=False, repr=False)

    def input_price(self, tokens: int) -> float:
        t: float = tokens / self._1m
        return self.prompt_price * t

    def output_price(self, tokens: int) -> float:
        t: float = tokens / self._1m
        return self.completion_price * t

    def image_price(self, n: int) -> float:
        return self.image_1080_1920_price * n

    def _calculate(self):
        self.input = self.input_price(self.prompt_tokens)
        self.output = self.output_price(self.completion_tokens)
        self.image = self.image_price(self.image_count)
        self.total = self.input + self.output + self.image

    def __str__(self):
        self._calculate()
        return super().__str__()

    def get_total(self):
        self._calculate()
        return self.total


@dataclass
class GPT4oPrice(_Pricing):
    prompt_price: float = field(default=2.5, init=False)
    completion_price: float = field(default=10.0, init=False)
    image_1080_1920_price: float = field(default=0.002763, init=False)


@dataclass
class GPT4oMiniPrice(_Pricing):
    prompt_price: float = field(default=0.150)
    completion_price: float = field(default=0.600)
    image_1080_1920_price: float = field(default=0.005525)


@dataclass
class WhisperPrice(_Pricing):
    stt: float = field(default=0.006)  # per minutes
    audio_length: int = field(default=0, init=False)

    def audio_price(self, sec: int) -> float:
        minutes = sec / 60
        return self.stt * minutes

    def _calculate(self):
        self.input = self.audio_price(self.audio_length)
