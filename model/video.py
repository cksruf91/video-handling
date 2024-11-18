from pathlib import Path
from typing import Tuple

import cv2

from model.image import ImageHandler


class _IterVideo:
    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture

    def __iter__(self):
        return self

    def __next__(self) -> ImageHandler:
        _available, frame = self.capture.read()
        if not _available:
            raise StopIteration('Done')
        return ImageHandler(frame)

    def release(self):
        self.capture.release()


class Video:
    def __init__(self, video: Path):
        self.cap = cv2.VideoCapture(str(video))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_channel = 3
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def iter_frame(self) -> _IterVideo:
        return _IterVideo(self.cap)

    @property
    def frame_size(self) -> Tuple[int, int, int]:
        return self._frame_width, self._frame_height, self._frame_channel

    @property
    def frame_count(self) -> int:
        return self._frame_count
