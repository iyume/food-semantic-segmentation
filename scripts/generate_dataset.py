from __future__ import annotations

import os
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Iterable, NamedTuple

import click
import cv2
import numpy as np

try:
    import foodseg as _
except ImportError:
    print("project is not installed")
    _parent_path = str(Path(__file__).parent.parent)
    sys.path.append(_parent_path)
    import foodseg as _

    sys.path.remove(_parent_path)

from foodseg.utils import classes, id2color


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


def _get_bbox_nonzero(img: np.ndarray) -> tuple[int, int, int, int]:
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


def food_loader(
    components: list[str], labels: list[int], check_overlap: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Read, compose boxed food and its label from universal components."""
    assert components and labels and len(components) == len(labels)
    imgs = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in components]
    assert all(i.shape == imgs[0].shape for i in imgs), "components must be same shape"
    assert all(
        i.ndim == 3 and i.shape[2] == 4 for i in imgs
    ), "components must be .png file"
    bbox = _get_bbox_nonzero(np.maximum.reduce([i[:, :, 3] for i in imgs]))
    bbox_slice = (slice(bbox[0], bbox[1]), slice(bbox[2], bbox[3]))
    food = np.zeros_like(imgs[0])
    label = np.zeros(imgs[0].shape[:2], dtype=np.uint8)
    for i in range(len(components)):
        # apply each component to result
        mask = imgs[i][:, :, 3] != 0
        # mask = mask & (label == 0)  # avoid overlap
        # some bug? or caused by resize?
        # if check_overlap:
        #     # check if components overlapped
        #     if not np.all(food[mask] == 0):
        #         warnings.warn(f"overlap at {components[i]} {np.average(food[mask])}")
        #         food2[mask] = [0, 0, 0, 255]
        #         cv2.imwrite("output.png", food2)
        food[mask] = imgs[i][mask]
        label[mask] = labels[i]
    return food[bbox_slice], label[bbox_slice]


def paste_image_on_alpha(
    source: np.ndarray,
    widget: np.ndarray,
    widget_label: np.ndarray,
    anchor: tuple[int, int],
    table_label: np.ndarray,
) -> np.ndarray:
    """Returns tuple of pasted image and label image."""
    assert source.shape[2] == 3 and widget.shape[2] == 4
    assert widget_label.ndim == 2
    assert widget.shape[:2] == widget_label.shape
    assert (
        source.shape[0] > widget.shape[0] + anchor[0]
        and source.shape[1] > widget.shape[1] + anchor[1]
    )
    source = source.copy()
    # label = np.zeros(source.shape[:2], dtype=np.uint8)
    mask = widget[:, :, 3] != 0
    h_s, w_s = anchor
    h_e, w_e = (h_s + widget.shape[0], w_s + widget.shape[1])
    source[h_s:h_e, w_s:w_e][mask] = widget[..., :3][mask]
    table_label[h_s:h_e, w_s:w_e] = widget_label  # remove mask to fix overlap
    # cv2.imwrite("output-label.png", translate_label(widget_label))
    # cv2.imwrite("output.png", translate_label(table_label))
    return source


def get_foods_data() -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Get food image (margin removed) and its class number.

    Directory Structure:
    |-- food name
        |-- 菜
            |-- 1.png
        |-- 盘子
            |-- 1.png
    """
    root = Path(food_dir)
    for food_name in root.iterdir():
        dish_path = food_name / "菜"
        plate_path = food_name / "盘子"
        assert dish_path.is_dir() and plate_path.is_dir()
        items = sorted(os.listdir(dish_path))
        for item in items:
            yield food_loader(
                [str(i / item) for i in (dish_path, plate_path)],
                [classes[food_name.name], classes["盘子"]],
            )


def get_tables_data() -> Iterable[tuple[np.ndarray, TableConfig]]:
    """Get table image and its config."""
    root = Path(table_dir)
    for item in root.iterdir():
        yield cv2.imread(str(item)), table_configs[item.stem]


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    assert img.ndim == 2 or img.ndim == 3
    # if abs(img.shape[0] - img.shape[1]) / min(img.shape[:2]) >= 0.15:
    #     ...
    width = int(height * (img.shape[1] / img.shape[0]))
    # NOTE: we MUST use INTER_NEAREST to not change the label broder
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)


def compress_to_height_batch(imgs: np.ndarray, height: int) -> np.ndarray:
    assert imgs.ndim == 4 or imgs.ndim == 3
    return np.stack([resize_to_height(i, height) for i in imgs])


def get_dataset(datanum: int = 20) -> tuple[np.ndarray, np.ndarray]:
    foods = list(get_foods_data())
    tables = list(get_tables_data())
    res_tables = []
    res_labels = []
    for _ in range(datanum):
        table, (_, food_size, anchors) = random.choice(tables)
        label = np.zeros(table.shape[:2], dtype=np.uint8)
        for anchor in anchors:
            food_img, food_class = random.choice(foods)
            food_img = resize_to_height(food_img, food_size[0])
            food_class = resize_to_height(food_class, food_size[0])
            table = paste_image_on_alpha(table, food_img, food_class, anchor, label)
        # label = np.maximum.reduce(labels)
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
            cv2.imwrite(str(labelvis_path / f"{i}.png"), translate_label(labels[i]))


def translate_label(label: np.ndarray) -> np.ndarray:
    """Convert label into colorized photo."""
    assert label.ndim == 2
    label_vis = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_num, color in id2color.items():
        label_vis[:, :, 0][label == class_num] = color[2]  # b channel
        label_vis[:, :, 1][label == class_num] = color[1]  # g channel
        label_vis[:, :, 2][label == class_num] = color[0]  # r channel
    return label_vis


@click.command()
@click.option("-n", "--number", default=40, type=click.INT)
@click.option(
    "-o", "--output-dir", default="dataset-compressed", type=click.Path(file_okay=False)
)
def main(number: int, output_dir: str):
    print("generating dataset...")
    imgs, labels = get_dataset(number)
    print("resizing...")
    imgs = compress_to_height_batch(imgs, 600)
    labels = compress_to_height_batch(labels, 600)
    if Path(output_dir).is_dir():
        print(f"cleaning {output_dir}")
        shutil.rmtree(output_dir)
    print(f"saved at {output_dir}")
    save_dataset(imgs, labels, output_dir)


if __name__ == "__main__":
    main()
