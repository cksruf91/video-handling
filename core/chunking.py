from collections import deque
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.model.image import ImageHandler
from core.model.video import Video
from utile.progress_bar import ProgressBar
from utile.time_utils import minutes_sec_formating


class VideoChunker:

    def __init__(self, video_file: Path, image_dir: Path):
        print('Grouping Frames')
        self.video = Video(video_file)
        self.image_dir = image_dir.joinpath('frames')
        self.image_dir.mkdir(exist_ok=True, parents=True)
        print(f'\tL video file : {video_file}')
        print(f'\tL save dir : {self.image_dir}')
        self.total_frames = self.video.frame_count
        self._min_sim = 0.7
        self.img_queue = deque()
        self.sim_queue = deque()

    def run(self):

        for jpg in self.image_dir.glob(pattern='*.jpg'):
            jpg.unlink()

        prev_frame = ImageHandler(np.uint8(
            np.random.randint(0, 255, self.video.frame_size)
        ))
        group_id, similarity = 0, 0.
        task = ProgressBar(self.video.iter_frame(), max_value=self.video.frame_count, bar_length=50, prefix='\t')
        for i, (frame, milli_sec) in enumerate(task):
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

            self.img_queue.append((frame, i, milli_sec))
            prev_frame = frame

        if len(self.img_queue) > 1:
            self.save_frame_group(self.img_queue, group_id)
        self.video.cap.release()

    def save_frame_group(self, q: deque[tuple[ImageHandler, int, float]], group_id: int) -> None:
        first = 0
        mid = len(q) // 2
        last = -1
        for idx in [first, mid, last]:
            save_frame, no, milli_sec = q[idx]
            position = minutes_sec_formating(milli_sec)
            save_file = self.image_dir.joinpath(f'frame_{group_id:04d}_{position}_{no:05d}.jpg')
            if not save_frame.write(save_file):
                raise Exception(f'failed to write {save_file.absolute()}')

    def confidence_limit(self, sims: deque[float]) -> float:
        if len(sims) < 2:
            return self._min_sim
        _mu = np.mean(sims)
        _s = np.std(sims)
        _n = max(len(sims), 10)
        score = _mu - (1.96 * _s / np.sqrt(_n))  # 신뢰수준 99%
        return max(score.item(), self._min_sim)
