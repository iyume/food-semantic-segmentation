# scripts that should be run through `python scripts/evaluate.py`
import cv2
import numpy as np
import torch
from torchvision import transforms

from .dataset_generator import id2color
from .models.unet import UNet

evalute_weight = "pretrained/model_epoch10.pth"
device = torch.device("cpu")

net = UNet(3, 12)
net.to(device)
net.load_state_dict(torch.load("pretrained/model_ep112.pth", device))
net.eval()

_transform = transforms.Compose([transforms.ToTensor()])


def transform(cv_img: np.ndarray) -> torch.Tensor:
    return _transform(cv_img)  # type: ignore


def evaluate(cv_img: np.ndarray) -> np.ndarray:
    """(BGR) -> BGR."""
    assert cv_img.ndim == 3
    assert cv_img.shape[2] == 3
    x = transform(cv_img)
    x.unsqueeze_(0)
    out: torch.Tensor = net(x)
    out = out.detach().cpu().squeeze().numpy()
    out = np.argmax(out, 0)
    out_vis = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for class_num, color in id2color.items():
        out_vis[:, :, 0][out == class_num] = color[2]  # b channel
        out_vis[:, :, 1][out == class_num] = color[1]  # g channel
        out_vis[:, :, 2][out == class_num] = color[0]  # r channel
    return out_vis


def test():
    testimg = cv2.imread("dataset-compressed/images/1.png")
    out_vis = evaluate(testimg)
    cv2.imwrite("output.png", out_vis)


if __name__ == "__main__":
    test()
