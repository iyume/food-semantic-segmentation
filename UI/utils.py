import os
import time
import typing as t
from functools import wraps

import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage

from .config import config

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


from foodseg.evaluate import evaluate as evaluate

evaluate = throttle(evaluate, config.evaluate_interval)
