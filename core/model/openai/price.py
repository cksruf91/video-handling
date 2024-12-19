from dataclasses import dataclass, field


class _Pricing:
    input_tokens: float = field(default=None)
    output_tokens: float = field(default=None)
    image_1080_1920: float = field(default=None)

    def input_price(self, tokens: int) -> float:
        n: float = tokens / 10000
        return self.input_tokens * n

    def output_price(self, tokens: int) -> float:
        n: float = tokens / 10000
        return self.output_tokens * n

    def image_price(self, n: int) -> float:
        return self.image_1080_1920 * n


@dataclass
class GPT4oPriceTag(_Pricing):
    input_tokens: float = field(default=2.5)
    output_tokens: float = field(default=10.0)
    image_1080_1920: float = field(default=0.002763)


@dataclass
class GPT4oMiniPriceTag(_Pricing):
    input_tokens: float = field(default=0.150)
    output_tokens: float = field(default=0.600)
    image_1080_1920: float = field(default=0.005525)


@dataclass
class WhisperPriceTag(_Pricing):
    stt: float = field(default=0.006)  # per minutes

    def audio_price(self, minutes: int) -> float:
        return self.stt * minutes

    def input_price(self, tokens: int) -> float:
        raise NotImplementedError("Whisper does not support input price")

    def output_price(self, tokens: int) -> float:
        raise NotImplementedError("Whisper does not support output price")

    def image_price(self, n: int) -> float:
        raise NotImplementedError("Whisper does not support image price")
