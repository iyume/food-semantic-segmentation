from contextlib import redirect_stdout
from typing import Callable, Union, cast

import cv2
import numpy as np
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap, QResizeEvent
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .config import config
from .utils import CooldownReject, array2QImage, evaluate, get_asset, min_size


def alert_info(msg: str) -> Callable[[], int]:
    msgbox = QMessageBox()
    msgbox.setIcon(QMessageBox.Information)
    msgbox.setWindowTitle("Info")
    msgbox.setText(msg)
    return lambda: msgbox.exec()


class VideoThread(QThread):
    # https://wiki.qt.io/QThreads_general_usage
    signal_read = pyqtSignal(np.ndarray)

    def __init__(self, device: Union[str, int]) -> None:
        super().__init__()
        self.device = device
        self.capture = cv2.VideoCapture(device)
        self.stopped = True

    def stop(self) -> None:
        """Stop and release camera."""
        self.stopped = True
        # not start or finished
        assert self.wait(300)
        assert not self.isRunning()
        self.capture.release()

    def run(self) -> None:
        # NOTE: QThread can't be debugged, so add test code
        self.stopped = False
        while True:
            if self.stopped:
                print(f"camera {self.device} stopped")
                return
            with redirect_stdout(None):
                # shadow the frame info
                ret, cv_img = self.capture.read()
            if ret:
                self.signal_read.emit(cv_img)
            else:
                self.stopped = True
                print(f"camera {self.device} not available")


class FlexImage(QLabel):
    def resizeEvent(self, a0: QResizeEvent) -> None:
        print(self.objectName(), self.size())
        return super().resizeEvent(a0)


class WebcamText(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.object_text = text

    def cam_opened(self) -> None:
        # emoji cannot display properly anyway
        # self.setText(self.object_text + " 🟢")
        self.setText(self.object_text + " √")
        "⚫"

    def cam_closed(self) -> None:
        # self.setText(self.object_text + " 🔴")
        self.setText(self.object_text + " x")

    def update_status(self, opened: bool) -> None:
        if opened:
            self.cam_opened()
        else:
            self.cam_closed()


class View(QWidget):
    min_image_size = QSize(200, 200)
    # when scaled KeepRatio, it refers to height
    # 900x600 -> 450x300

    def __init__(self):
        super().__init__()
        self.started = False

        self.camera_device = config.device

        self.setWindowTitle("Real-time Food Recognition")
        self.setWindowIcon(QIcon(get_asset("project-sekai-icon.jpg")))
        self.setup_layout()
        self.setup_default_image()

        self.load_camera()
        # 0.0 when webcam unavailable
        self.camera_thread.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    def setup_layout(self) -> None:
        # State Principle: Only Set Widget
        mainbox = QHBoxLayout()
        cambox = QVBoxLayout()
        outbox = QVBoxLayout()
        mainbox.addLayout(cambox)
        mainbox.addLayout(outbox)
        self.in_image = in_image = FlexImage()
        in_image.setObjectName("input image")
        in_image.setMinimumSize(self.min_image_size)
        self.in_image.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        # self.in_image.setScaledContents(True)  # auto scaled on resize
        self.in_label = in_label = WebcamText("Webcam")
        cambox.addWidget(in_image)
        cambox.addWidget(in_label)
        self.out_image = out_image = FlexImage()
        out_image.setObjectName("output image")
        out_image.setMinimumSize(self.min_image_size)
        self.out_image.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        # self.out_image.setScaledContents(True)
        self.out_label = out_label = QLabel("Output")
        outbox.addWidget(out_image)
        outbox.addWidget(out_label)

        buttonsbox = QHBoxLayout()
        self.bt_refresh = bt_refresh = QPushButton("Refresh")
        self.bt_start = bt_start = QPushButton("Start")
        self.bt_stop = bt_stop = QPushButton("Stop")
        self.bt_fullscreen = bt_fullscreen = QPushButton("FullScreen")
        buttonsbox.addStretch(1)
        buttonsbox.addWidget(bt_refresh)
        buttonsbox.addWidget(bt_start)
        buttonsbox.addWidget(bt_stop)
        buttonsbox.addWidget(bt_fullscreen)
        bt_refresh.clicked.connect(self.handle_refresh)
        bt_start.clicked.connect(self.handle_start)
        bt_stop.clicked.connect(self.handle_stop)
        bt_fullscreen.clicked.connect(self.handle_fullscreen)

        layout = QVBoxLayout()
        layout.addLayout(mainbox)
        layout.addLayout(buttonsbox)
        self.setLayout(layout)

    def setup_default_image(self) -> None:
        placeholder = QImage(get_asset("elementor-placeholder-image.jpg"))
        self.in_image.setPixmap(
            QPixmap.fromImage(placeholder.scaledToHeight(min_size(self.min_image_size)))
        )
        self.out_image.setPixmap(
            QPixmap.fromImage(placeholder.scaledToHeight(min_size(self.min_image_size)))
        )

    def load_camera(self) -> None:
        # establish new connection to webcam
        self.camera_thread = VideoThread(self.camera_device)
        self.in_label.update_status(self.camera_thread.capture.isOpened())
        self.camera_thread.signal_read.connect(self.handle_camera_receive)
        # TODO: start should be called after window shown
        self.camera_thread.start()

    def handle_refresh(self) -> None:
        if self.camera_thread.capture.isOpened():
            alert_info("camera already opened")()
            return
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.load_camera()

    def handle_start(self) -> None:
        if not self.camera_thread.capture.isOpened():
            alert_info("camera not found")()
        else:
            self.started = True

    def handle_stop(self) -> None:
        self.started = False
        View.min_image_size = QSize(
            View.min_image_size.height() + 200, View.min_image_size.height() + 200
        )
        self.setup_default_image()

    def handle_fullscreen(self) -> None:
        self.out_image.setWindowFlags(
            cast(
                Qt.WindowType, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
            )
        )
        self.out_image.setScaledContents(True)
        self.out_image.showFullScreen()

    def handle_camera_receive(self, cv_img: np.ndarray) -> None:
        """Receive camera signal and update image."""
        assert cv_img.ndim == 3
        assert cv_img.shape[2] == 3
        pixmap = self._cv2pixmap(cv_img)
        self.in_image.setPixmap(pixmap)
        if self.started:
            try:
                cv_img_colormap = evaluate(cv_img)
            except CooldownReject:
                pass
            else:
                pixmap = self._cv2pixmap(cv_img_colormap)
                self.out_image.setPixmap(pixmap)

    def _cv2pixmap(self, cv_img: np.ndarray) -> QPixmap:
        qt_img = array2QImage(cv_img)
        qt_img = qt_img.scaledToHeight(self.in_image.height())
        return QPixmap.fromImage(qt_img)
