from dataclasses import dataclass, field


@dataclass
class Pricing:
    prompt_price: float = field(default=0, init=False)
    cached_prompt_price: float = field(default=0, init=False)
    completion_price: float = field(default=0, init=False)
    image_1080_1920_price: float = field(default=0, init=False)
    audio_price: float = field(default=0, init=False)

    _1m: int = field(default=1000000, init=False, repr=False)


@dataclass
class GPT4oPrice(Pricing):
    prompt_price: float = field(default=2.5, init=False)
    cached_prompt_price: float = field(default=1.25, init=False)
    completion_price: float = field(default=10.0, init=False)
    image_1080_1920_price: float = field(default=0.002763, init=False)


@dataclass
class GPT4oMiniPrice(Pricing):
    prompt_price: float = field(default=0.150)
    cached_prompt_price: float = field(default=0.075, init=False)
    completion_price: float = field(default=0.600)
    image_1080_1920_price: float = field(default=0.005525)


@dataclass
class WhisperPrice(Pricing):
    audio_price: float = field(default=0.006)  # per minutes
