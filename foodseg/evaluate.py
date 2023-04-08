# scripts that should be run through `python -m foodseg.evaluate`
import click
import cv2
import numpy as np
import torch
from torchvision import transforms

from .dataset_generator import id2color
from .models.unet import UNet
from .utils import classes

device = torch.device("cpu")

net = UNet(3, len(classes))
net.to(device)
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


@click.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("-p", "--pth-file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default="output.png", type=click.STRING)
def cli(image: str, pth_file: str, output: str):
    net.load_state_dict(torch.load(pth_file, device)["model_state_dict"])
    out = evaluate(cv2.imread(image))
    cv2.imwrite(output, out)


if __name__ == "__main__":
    cli()
