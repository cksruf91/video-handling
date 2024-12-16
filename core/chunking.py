from collections import deque
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.model.image import ImageHandler
from core.model.video import Video
from utile.progress_bar import ProgressBar
from utile.utils import minutes_sec_formating, softmax


class VideoChunker:

    def __init__(self, video_file: Path, image_dir: Path):
        print('Grouping Frames')
        self.video = Video(video_file)
        self.image_dir = image_dir.joinpath('frames')
        self.image_dir.mkdir(exist_ok=True, parents=True)
        print(f'\tL video file : {video_file}')
        print(f'\tL save dir : {self.image_dir}')
        w, h, c = self.video.frame_size
        self.target_size = (1080, 1920) if h > w else (1920, 1080)
        print(f'\tL frame_size : {self.video.frame_size}')

        self.img_queue = deque()
        self.sim_queue = deque()

    def run(self):

        for jpg in self.image_dir.glob(pattern='*.jpg'):
            jpg.unlink()

        prev_frame, prev_sim = None, None
        group_id = 0
        compute_size = [v//2 for v in self.target_size]
        task = ProgressBar(self.video.iter_frame(), max_value=self.video.frame_count, bar_length=50, prefix='\t')

        # history = Path('temp/sims.csv')  # -------------------------------------------------------------------------
        # history.open('w').write('frame,similarity,confidence\n')  # ------------------------------------------------
        for i, (frame, milli_sec) in enumerate(task):
            frame: ImageHandler
            if prev_frame is None:
                prev_frame = frame.copy()
            current_frame = frame.copy().resize(*compute_size).grayscale().flat()
            prev_frame = prev_frame.copy().resize(*compute_size).grayscale().flat()
            similarity = cosine_similarity([current_frame], [prev_frame])[0][0]
            if prev_sim is None:
                prev_sim = similarity
            _diff = abs(similarity - prev_sim)

            task.update(suffix=f"{i}/{self.video.frame_count} frames / sim:{similarity:0.5f}")
            _confidence = self.confidence_limit(self.sim_queue)
            # history.open('a').write(f'{i},{_diff},{_confidence}\n')  # ---------------------------------------------

            if _diff > _confidence:
                self.sim_queue.clear()
                if len(self.img_queue) > 1:
                    self.save_frame_group(self.img_queue, group_id)
                self.img_queue.clear()
                group_id += 1
            else:
                self.sim_queue.append(_diff)
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
            if not save_frame.resize(*self.target_size).write(save_file):
                raise Exception(f'failed to write {save_file.absolute()}')

    @staticmethod
    def confidence_limit(sims: deque[float]) -> float:
        if not sims:
            return 1
        sims = list(sims)[-30:]
        if len(sims) < 2:
            return sims[-1] * 2
        w = softmax(len(sims))
        _mu: float = sum([v * w for v, w in zip(sims, w)])  # 가중 평균
        _s = np.std(sims)
        score = _mu + (6 * _s)
        return max(score, 0.001)
