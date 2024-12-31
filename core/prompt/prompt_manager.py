from pathlib import Path

import yaml


class PromptManager:
    _FILE = Path('.').joinpath('resource').joinpath('prompt.yaml')
    _PROMPT = yaml.safe_load(_FILE.open('r'))
    _SUMMARY = _PROMPT.get('app').get('summary').get('v3')

    SUMMARY_SYSTEM = _SUMMARY.get('sys_prompt')
    SUMMARY_USER = _SUMMARY.get('user_prompt')
    SUMMARY_ASSIST = _SUMMARY.get('assistant_prompt')

    CAPTIONING = _PROMPT.get('app').get('captioning').get('v3').get('prompt')

    KEYWORD_SYSTEM = _PROMPT.get('app').get('keyword').get('v1').get('system_prompt')
    KEYWORD_USER = _PROMPT.get('app').get('keyword').get('v1').get('user_prompt')
