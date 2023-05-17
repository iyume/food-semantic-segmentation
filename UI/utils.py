import os
import time
import typing as t
from functools import wraps

import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage

from foodseg.utils import classes

from . import config

_pathnorm = os.path.normpath

TC = t.TypeVar("TC", bound=t.Callable)


def array2QImage(x: np.ndarray) -> QImage:
    """Convert BGR image (non-alpha) to QImage."""
    assert x.ndim == 3
    assert x.shape[2] in (1, 3)
    h, w, c = x.shape
    return QImage(x.data, w, h, c * w, QImage.Format.Format_BGR888)


def fixpyqtSlot(func: t.Any) -> t.Callable[..., None]:
    # fix lower version of PyQt5 pyqtSlot stub error
    return func


def get_asset(filename: str) -> str:
    """Returns full filename at assets directory."""
    return os.path.join(_pathnorm(config.assets_dir), _pathnorm(filename))


def pred_to_output(source: np.ndarray, gt_pred: np.ndarray) -> np.ndarray:
    """Generate output from pred. Parameter source and gt_pred must have same size."""
    assert source.ndim == 3, gt_pred.ndim == 2
    assert source.shape[:2] == gt_pred.shape[:2]
    assert source.shape[2] == 3
    output = np.zeros_like(source)
    # output_mask = (gt_pred != classes["背景"]) & (gt_pred != classes["盘子"])
    output_mask = np.any([gt_pred == c for c in range(1, 12)], axis=0)
    output[output_mask] = source[output_mask]
    return output


def min_size(size: QSize) -> int:
    return min(size.height(), size.width())


def max_size(size: QSize) -> int:
    return max(size.height(), size.width())


class CooldownReject(RuntimeError):
    ...


def throttle(func: TC, sec: float) -> TC:
    prev = 0.0

    @wraps(func)
    def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        nonlocal prev
        now = time.time()
        if now - prev > sec:
            prev = time.time()
            return func(*args, **kwargs)
        raise CooldownReject

    # cast of ParamSpec
    return t.cast(TC, wrapper)


from foodseg.evaluate import Evaluator

evaluate = throttle(
    Evaluator(pth_file=config.pth_file, device=config.evaluate_device).evaluate,
    config.evaluate_interval,
)
