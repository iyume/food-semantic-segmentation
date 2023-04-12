import itertools
import os
import random
from collections import UserList
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import cv2
import numpy as np

T = TypeVar("T")

image_dir = Path("dataset-compressed/images")
label_dir = Path("dataset-compressed/visualizes")


def read_images(path: Path) -> np.ndarray:
    res = []
    for p in path.iterdir():
        res.append(cv2.imread(str(p)))
    return np.stack(res)


def _repeat(itor: Iterable[T], intervals: Iterable[int]) -> Iterator[T]:
    # turns (img1, img2) (3, 4)
    # into (img1,img1,img1,img2,img2,img2,img2)
    return itertools.chain.from_iterable(
        itertools.repeat(obj, val) for obj, val in zip(itor, intervals)
    )


class _ExtensiveIterable(UserList[T]):
    """Effective extensible container from iterable (infinite maybe)."""

    def __init__(self, __iterable: Iterable[T]) -> None:
        super().__init__()
        self.iterator = iter(__iterable)

    def extensive(self, maxlen: int) -> None:
        if maxlen > len(self):
            self.extend(itertools.islice(self.iterator, maxlen - len(self)))


class Processor:
    def __init__(self, min_interval: int, max_interval: int) -> None:
        assert 0 < min_interval < max_interval
        # generate random parameters
        # NOTE: intervals should be definite
        self.intervals = _ExtensiveIterable(
            random.randint(min_interval, max_interval) for _ in itertools.count()
        )
        # self.indexier = itertools.chain.from_iterable(
        #     itertools.repeat(i, val) for i, val in enumerate(self.intervals)
        # )

    def write(
        self,
        filename: str,
        img_iterator: Iterator[np.ndarray],
        video_sec: int = 120,
    ) -> None:
        # create video with random-ranged interval
        assert filename.endswith(".mp4")  # maybe lowercase
        self.intervals.extensive(video_sec)  # hackway impl
        img_repeater = _repeat(img_iterator, self.intervals)
        prev_image = next(img_repeater, None)
        if prev_image is None:
            raise RuntimeError("image iterator is empty") from None
        size = prev_image.shape[1], prev_image.shape[0]
        video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MP4V"), 1, size)
        video.write(prev_image)
        for _ in range(video_sec):
            try:
                prev_image = next(img_repeater)
            except StopIteration:
                raise RuntimeError("images not enough") from None
            video.write(prev_image)


def _get_image_iterator(path: Path) -> Iterator[np.ndarray]:
    return (cv2.imread(str(path / i)) for i in sorted(os.listdir(path)))


processor = Processor(3, 7)
processor.write("video.mp4", _get_image_iterator(image_dir))
processor.write("video-label.mp4", _get_image_iterator(label_dir))
