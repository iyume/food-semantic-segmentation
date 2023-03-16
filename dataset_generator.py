from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, NamedTuple

import cv2
import numpy as np


class TableConfig(NamedTuple):
    size: tuple[int, int]
    food_size: tuple[int, int]  # square
    anchors: list[tuple[int, int]]


food_dir = "foods"
table_dir = "tables"
# food_size is at least 200 to paste
table_configs = {
    "table1": TableConfig(
        size=(1050, 1400),
        food_size=(200, 200),
        anchors=[(330, 360), (540, 360), (330, 860), (540, 860)],
    )
}
create_visualizes = True
id2color = {
    1: (150, 0, 250),
    2: (200, 50, 250),
    3: (30, 100, 250),
    4: (30, 0, 200),
    5: (150, 50, 200),
    6: (200, 100, 200),
    7: (30, 0, 150),
    8: (30, 50, 150),
    9: (150, 100, 150),
    10: (200, 0, 100),
    11: (30, 50, 100),
}  # id: (R, G, B)
classes = {
    "白切鸡": 1,
    "炒西兰花": 2,
    "干煸豆子": 3,
    "红烧茄子": 4,
    "红烧鱼": 5,
    "家常豆腐": 6,
    "尖椒肉丝": 7,
    "凉拌黄瓜": 8,
    "土豆牛肉": 9,
    "土豆丝": 10,
    "西红柿鸡蛋": 11,
}


def _get_bbox_nonzero(img: np.ndarray):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.argwhere(rows).squeeze()[[0, -1]]
    cmin, cmax = np.argwhere(cols).squeeze()[[0, -1]]
    return rmin, rmax, cmin, cmax


def imread_crop_margin(path: str) -> np.ndarray:
    assert path.endswith(".png"), f"{path} must be png file with alpha channel"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    alpha_channel = img[:, :, 3]
    bbox = _get_bbox_nonzero(alpha_channel)
    img = img[bbox[0] : bbox[1], bbox[2] : bbox[3]]
    return img


def paste_image_on_alpha(
    source: np.ndarray,
    to_paste: np.ndarray,
    anchor: tuple[int, int] = (0, 0),
    class_num: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns tuple of pasted image and label image."""
    assert source.shape[2] == 3 and to_paste.shape[2] == 4
    assert (
        source.shape[0] > to_paste.shape[0] + anchor[0]
        and source.shape[1] > to_paste.shape[1] + anchor[1]
    )
    source = source.copy()
    label = np.zeros(source.shape[:2], dtype=np.uint8)
    alpha_channel = to_paste[:, :, 3]
    h_s, w_s = anchor
    h_e, w_e = (h_s + to_paste.shape[0], w_s + to_paste.shape[1])
    label[h_s:h_e, w_s:w_e][alpha_channel != 0] = class_num
    source[label == class_num] = to_paste[..., :3][alpha_channel != 0]
    return source, label


def get_foods_data() -> Iterable[tuple[np.ndarray, int]]:
    """Get food image (margin removed) and its class number."""
    root = Path(food_dir)
    for path in root.iterdir():
        for item in path.iterdir():
            yield imread_crop_margin(str(item)), classes[path.name]


def get_tables_data() -> Iterable[tuple[np.ndarray, TableConfig]]:
    """Get table image and its config."""
    root = Path(table_dir)
    for item in root.iterdir():
        yield cv2.imread(str(item)), table_configs[item.stem]


def resize_flexible_width(img: np.ndarray, height: int) -> np.ndarray:
    assert img.ndim == 2 or img.ndim == 3
    # if abs(img.shape[0] - img.shape[1]) / min(img.shape[:2]) >= 0.15:
    #     ...
    width = int(height * (img.shape[1] / img.shape[0]))
    return cv2.resize(img, (width, height))


def compress_batch(imgs: np.ndarray, height: int) -> np.ndarray:
    assert imgs.ndim == 4 or imgs.ndim == 3
    return np.stack([resize_flexible_width(i, height) for i in imgs])


def get_dataset(datanum: int = 20) -> tuple[np.ndarray, np.ndarray]:
    foods = list(get_foods_data())
    tables = list(get_tables_data())
    res_tables = []
    res_labels = []
    for _ in range(datanum):
        table, (_, food_size, anchors) = random.choice(tables)
        labels = []
        for anchor in anchors:
            food_img, food_class = random.choice(foods)
            food_img = resize_flexible_width(food_img, food_size[0])
            table, label = paste_image_on_alpha(table, food_img, anchor, food_class)
            labels.append(label)
        label = np.stack(labels).max(0)
        res_tables.append(table)
        res_labels.append(label)
    return np.stack(res_tables), np.stack(res_labels)


def save_dataset(
    imgs: np.ndarray, labels: np.ndarray, output_dir: str = "dataset"
) -> None:
    # imgs (N,H,W,C). labels (N,H,W)
    assert imgs.ndim == 4 and labels.ndim == 3
    path = Path(output_dir)
    image_path = path / "images"
    label_path = path / "labels"
    labelvis_path = path / "visualizes"
    image_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)
    if create_visualizes:
        labelvis_path.mkdir(parents=True, exist_ok=True)
    for i in range(len(imgs)):
        cv2.imwrite(str(image_path / f"{i}.png"), imgs[i])
        cv2.imwrite(str(label_path / f"{i}.png"), labels[i])
        if create_visualizes:
            label = labels[i]
            label_vis = np.zeros((*label.shape, 3), dtype=np.uint8)
            for class_num, color in id2color.items():
                label_vis[:, :, 0][label == class_num] = color[2]  # b channel
                label_vis[:, :, 1][label == class_num] = color[1]  # g channel
                label_vis[:, :, 2][label == class_num] = color[0]  # r channel
            cv2.imwrite(str(labelvis_path / f"{i}.png"), label_vis)


if __name__ == "__main__":
    imgs, labels = get_dataset(400)
    imgs = compress_batch(imgs, 300)
    labels = compress_batch(labels, 300)
    save_dataset(imgs, labels, "dataset-compressed")
