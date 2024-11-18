from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from model.image import ImageHandler
from model.video import Video
from utile.progress_bar import ProgressBar


class VideoChunker:

    def __init__(self, video_file: Path, save_dir: Path):
        self.video = Video(video_file)
        self.save_dir = save_dir
        print(f'save dir : {self.save_dir.absolute()}')
        self.total_frames = self.video.frame_count
        self._min_sim = 0.7
        self.img_queue = deque()
        self.sim_queue = deque()

    def run(self):

        for jpg in self.save_dir.glob(pattern='*.jpg'):
            jpg.unlink()

        prev_frame = ImageHandler(np.uint8(
            np.random.randint(0, 255, self.video.frame_size)
        ))
        group_id, similarity = 0, 0.
        task = ProgressBar(self.video.iter_frame(), max_value=self.video.frame_count, bar_length=50)
        for i, frame in enumerate(task):
            frame: ImageHandler
            current_frame = frame.copy().resize(500, 250).grayscale().flat()
            prev_frame = prev_frame.copy().resize(500, 250).grayscale().flat()
            similarity = cosine_similarity([current_frame], [prev_frame])[0][0]
            task.update(suffix=f"{i}/{self.video.frame_count} frames / sim:{similarity:0.5f}")

            _confidence = self.confidence_limit(self.sim_queue)

            if similarity < _confidence:
                self.sim_queue.clear()
                if len(self.img_queue) > 1:
                    self.save_frame_group(self.img_queue, group_id)
                self.img_queue.clear()
                group_id += 1
                self.sim_queue.append(self._min_sim)
            else:
                self.sim_queue.append(similarity)

            self.img_queue.append((frame, i, similarity))
            prev_frame = frame

        if len(self.img_queue) > 1:
            self.save_frame_group(self.img_queue, group_id)
        self.video.cap.release()

    def save_frame_group(self, q: deque, group_id: int) -> None:
        mid = len(q) // 2
        for save_frame, no, sim in [q[0], q[mid], q[-1]]:
            save_file = self.save_dir.joinpath(f'frame_{group_id:04d}_{no:05d}.jpg')
            if not save_frame.write(save_file):
                raise Exception(f'failed to write {save_file.absolute()}')

    def confidence_limit(self, sims: Any) -> float:
        if len(sims) < 2:
            return self._min_sim
        _mu = np.mean(sims)
        _s = np.std(sims)
        _n = max(len(sims), 10)
        score = _mu - (1.96 * _s / np.sqrt(_n))  # 신뢰수준 99%
        return max(score.item(), self._min_sim)
