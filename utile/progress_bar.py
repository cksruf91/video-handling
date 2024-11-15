from typing import Iterable

import progressbar


class ProgressBar:
    widget = [
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]

    def __init__(self, *args, **kwargs):
        # self._bar = progressbar.ProgressBar(*args, **kwargs)
        raise RuntimeError('should initialize by {init} method')

    @classmethod
    def init(cls, iterable: Iterable):
        return progressbar.progressbar(iterable, widgets=cls.widget)
