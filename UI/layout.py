from typing import Callable, cast

import cv2
import numpy as np
from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from backend import evaluate

from .config import config
from .utils import array2QImage, get_asset


def alert_info(msg: str) -> Callable[[], int]:
    msgbox = QMessageBox()
    msgbox.setIcon(QMessageBox.Information)
    msgbox.setWindowTitle("Info")
    msgbox.setText(msg)
    return lambda: msgbox.exec()


class VideoThread(QThread):
    signal_read = pyqtSignal(np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        self.webcam = cv2.VideoCapture(config.device)

    def run(self) -> None:
        cap = self.webcam
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.signal_read.emit(cv_img)


class View(QWidget):
    image_size = QSize(300, 300)
    # when scaled KeepRatio, it refers to height
    # 900x600 -> 450x300

    def __init__(self):
        super().__init__()
        self.started = False

        self.camera_thread = VideoThread()
        self.camera_thread.signal_read.connect(self.handle_camera)

        self.setWindowTitle("Real-time Food Recognition")
        self.setWindowIcon(QIcon(get_asset("project-sekai-icon.jpg")))
        self.setup_layout()
        self.setup_default_image()

    def show(self) -> None:
        super().show()
        # self.camera_thread.start()

    def setup_layout(self) -> None:
        # State Principle: Only Set Widget
        mainbox = QHBoxLayout()
        cambox = QVBoxLayout()
        outbox = QVBoxLayout()
        mainbox.addLayout(cambox)
        mainbox.addLayout(outbox)
        self.in_image = in_image = QLabel()
        # self.in_image.setScaledContents(True)  # auto scaled on resize
        self.in_label = in_label = QLabel("Webcam")
        cambox.addWidget(in_image)
        cambox.addWidget(in_label)
        self.out_image = out_image = QLabel()
        # self.out_image.setScaledContents(True)
        self.out_label = out_label = QLabel("Output")
        outbox.addWidget(out_image)
        outbox.addWidget(out_label)

        buttonsbox = QHBoxLayout()
        self.button1 = button1 = QPushButton("Start")
        self.button2 = button2 = QPushButton("Stop")
        self.button3 = button3 = QPushButton("FullScreen")
        buttonsbox.addStretch(1)
        buttonsbox.addWidget(button1)
        buttonsbox.addWidget(button2)
        buttonsbox.addWidget(button3)
        button1.clicked.connect(self.handle_start)
        button2.clicked.connect(self.handle_stop)
        button3.clicked.connect(self.handle_fullscreen)

        layout = QVBoxLayout()
        layout.addLayout(mainbox)
        layout.addLayout(buttonsbox)
        self.setLayout(layout)

    def setup_default_image(self) -> None:
        placeholder = QImage(get_asset("elementor-placeholder-image.jpg"))
        self.in_image.setPixmap(
            QPixmap.fromImage(
                placeholder.scaled(self.image_size, Qt.AspectRatioMode.KeepAspectRatio)
            )
        )
        self.out_image.setPixmap(
            QPixmap.fromImage(
                placeholder.scaled(self.image_size, Qt.AspectRatioMode.KeepAspectRatio)
            )
        )

    def handle_start(self) -> None:
        self.started = True
        if not self.camera_thread.webcam.isOpened():
            alert_info("camera not found")()

    def handle_stop(self) -> None:
        self.started = False

    def handle_fullscreen(self) -> None:
        self.out_image.setWindowFlags(
            cast(
                Qt.WindowType, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
            )
        )
        self.out_image.setScaledContents(True)
        self.out_image.showFullScreen()

    def handle_camera(self, cv_img: np.ndarray) -> None:
        """Receive camera signal and update image."""
        assert cv_img.ndim == 3
        assert cv_img.shape[2] == 3
        pixmap = self._cv2pixmap(cv_img)
        self.in_image.setPixmap(pixmap)
        if self.started:
            cv_img_logits = evaluate(cv_img)
            pixmap = self._cv2pixmap(cv_img_logits)
            self.out_image.setPixmap(pixmap)

    def _cv2pixmap(self, cv_img: np.ndarray) -> QPixmap:
        qt_img = array2QImage(cv_img)
        qt_img = qt_img.scaled(self.image_size, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(qt_img)
