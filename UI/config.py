from typing import Tuple, Union

assets_dir: str = "UI/assets"

capture_target: Union[str, int] = "video.mp4"
"""Argument for `cv2.VideoCapture`."""

capture_image_size: Tuple[int, int] = (800, 600)
"""Capture image size (W,H). Useful for reduce evaluation time."""

evalute_output_image_size: Tuple[int, int] = (1080, 720)
"""Evaluation output image size (W,H)."""

evaluate_interval: float = 3
"""Evaluation cooldown interval."""

pth_file: str = "pretrained/model_v1.0_epoch350.pth"
"""Model pth file."""

evaluate_device: str = "cpu"
"""Model evaluation device."""
