from typing import Union

assets_dir: str = "UI/assets"

capture_target: Union[str, int] = "video.mp4"
"""Argument for `cv2.VideoCapture`."""

evaluate_interval: float = 10
"""Evaluation cooldown interval."""

pth_file: str = "pretrained/model_v1.0_epoch350.pth"
"""Model pth file."""

evaluate_device: str = "cpu"
"""Model evaluation device."""
