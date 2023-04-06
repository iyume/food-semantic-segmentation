from typing import Union


class Config:
    assets_dir: str = "UI/assets"

    device: Union[str, int] = "timer-5min.mp4"
    """Argument for `cv2.VideoCapture`."""

    evaluate_interval: float = 10


config = Config()
