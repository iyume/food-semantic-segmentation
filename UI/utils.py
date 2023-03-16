import os
import typing as t

import numpy as np
from PyQt5.QtGui import QImage

from .config import config

_pathnorm = os.path.normpath


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
