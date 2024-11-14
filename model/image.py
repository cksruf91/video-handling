import base64
import io
from copy import deepcopy
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from PIL.ImageFile import ImageFile


class ImageHandler:
    def __init__(self, data: Union[Path, ImageFile, np.ndarray] = None):
        if isinstance(data, Path):
            self._image: np.ndarray = cv2.imread(str(data))
            if self._image is None:
                raise ValueError("fail to read image")
        elif isinstance(data, ImageFile):
            self._image: np.ndarray = cv2.cvtColor(np.array(data), cv2.COLOR_BGR2RGB)
        elif isinstance(data, np.ndarray):
            self._image: np.ndarray = data
        else:
            raise ValueError(f"Data type {type(data)} not supported")

    @property
    def image(self) -> np.ndarray:
        return self._image

    def cvt_color(self, method: int):
        self._image = cv2.cvtColor(self._image, method)
        return self

    def copy(self):
        return deepcopy(self)

    def resize(self, width, height):
        self._image = cv2.resize(self._image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        return self

    def encoding(self) -> str:
        _, jpg_img = cv2.imencode('.jpg', self._image)
        return base64.b64encode(jpg_img).decode('utf-8')

    @classmethod
    def from_base64(cls, b64_string: str):
        imgdata = base64.b64decode(b64_string.encode('utf-8'))
        byte_image = io.BytesIO(imgdata)
        img = Image.open(byte_image)
        return cls(img)

    def flat(self):
        return self._image.flatten() / 255

    def write(self, path: Path) -> bool:
        return cv2.imwrite(str(path), self._image)

    def hist_eq(self):
        self._image = cv2.equalizeHist(self._image)
        return self
