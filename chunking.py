from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from model.image import ImageHandler
from model.video import Video


class VideoChunker:

    def __init__(self, video: Path, save_dir: Path):
        self.video = Video(str(video))
        self.save_dir = save_dir
        print(f'save dir : {self.save_dir.absolute()}')
        self.total_frames = self.video.frame_count
        self._min_sim = 0.7

    def run(self):

        for jpg in self.save_dir.glob(pattern='*.jpg'):
            jpg.unlink()

        prev_frame = ImageHandler(np.uint8(
            np.random.randint(0, 255, self.video.frame_size)
        ))
        group_id, similarity = 0, 0.
        img_queue = deque()
        sim_queue = deque()
        for i, frame in enumerate(self.video.iter_frame()):
            frame: ImageHandler
            current_frame = frame.copy().resize(500, 250).cvt_color(cv2.COLOR_BGR2GRAY).flat()
            prev_frame = prev_frame.copy().resize(500, 250).cvt_color(cv2.COLOR_BGR2GRAY).flat()
            similarity = cosine_similarity([current_frame], [prev_frame])[0][0]

            _confidence = self.confidence(sim_queue)

            if similarity < _confidence:
                sim_queue.clear()
                if len(img_queue) > 1:
                    self.save_frame_group(img_queue, group_id, similarity)
                img_queue.clear()
                group_id += 1
                sim_queue.append(self._min_sim)
            else:
                sim_queue.append(similarity)

            img_queue.append((frame, i))
            prev_frame = frame

        if len(img_queue) > 1:
            self.save_frame_group(img_queue, group_id, similarity)
        self.video.cap.release()

    def save_frame_group(self, q: deque, group_id: int, sim: float) -> None:
        mid = len(q) // 2
        for save_frame, no in [q[0], q[mid], q[-1]]:
            save_file = self.save_dir.joinpath(f'frame_{group_id:04d}_{no:05d}.jpg')
            if not save_frame.write(save_file):
                raise Exception(f'failed to write {save_file.absolute()}')
        print(f"{no}/{self.total_frames}-frame sim:{sim}")

    def confidence(self, sims: Any) -> float:
        if len(sims) < 2:
            return self._min_sim
        _mu = np.mean(sims)
        _s = np.std(sims)
        _n = max(len(sims), 10)
        score = _mu - (2.58 * _s / np.sqrt(_n))  # 신뢰수준 95%
        return max(score.item(), self._min_sim)
