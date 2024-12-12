from pathlib import Path

import yaml


class PromptManager:
    _FILE = Path('.').joinpath('core').joinpath('resource').joinpath('prompt.yaml')
    _PROMPT = yaml.safe_load(_FILE.open('r'))

    SUMMARY = _PROMPT.get('app').get('summary').get('prompt')
    CAPTIONING = _PROMPT.get('app').get('captioning').get('prompt')