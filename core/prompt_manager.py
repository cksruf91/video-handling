from pathlib import Path

import yaml


class PromptManager:
    _FILE = Path('.').joinpath('core').joinpath('resource').joinpath('prompt.yaml')
    _PROMPT = yaml.safe_load(_FILE.open('r'))

    SUMMARY = _PROMPT.get('app').get('summary').get('v1').get('prompt')
    CAPTIONING = _PROMPT.get('app').get('captioning').get('v1').get('prompt')
    KEYWORD_SYSTEM = _PROMPT.get('app').get('keyword').get('v1').get('system_prompt')
    KEYWORD_USER = _PROMPT.get('app').get('keyword').get('v1').get('user_prompt')
