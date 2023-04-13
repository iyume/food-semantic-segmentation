# scripts that should be run through `python -m foodseg.evaluate`
from typing import ClassVar, Optional, cast

import click
import cv2
import numpy as np
import torch
import time

from .datasets import GeneratedDataset
from .models.unet import UNet
from .utils import State, classes, id2color

transform = GeneratedDataset.transform


class Evaluator:
    model: ClassVar = UNet(3, len(classes))
    model.eval()

    def __init__(self, pth_file: Optional[str] = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model.to(self.device)
        if pth_file is not None:
            state = cast(State, torch.load(pth_file, self.device))
            self.model.load_state_dict(state["model_state_dict"])

    def evaluate(self, cv_img: np.ndarray, timeit: bool = False) -> np.ndarray:
        """(BGR) -> BGR."""
        assert cv_img.ndim == 3
        assert cv_img.shape[2] == 3
        stime = time.time()
        x = transform(cv_img)
        x.unsqueeze_(0)
        out: torch.Tensor = self.model(x)
        out = out.detach().cpu().squeeze().numpy()
        out = np.argmax(out, 0)
        out_vis = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
        for class_num, color in id2color.items():
            out_vis[:, :, 0][out == class_num] = color[2]  # b channel
            out_vis[:, :, 1][out == class_num] = color[1]  # g channel
            out_vis[:, :, 2][out == class_num] = color[0]  # r channel
        if timeit:
            print(f"evaluation time: {time.time() - stime:.6f}s")
        return out_vis


@click.command()
@click.argument("imgfile", type=click.Path(exists=True, dir_okay=False))
@click.option("-p", "--pth-file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default="output.png", type=click.STRING)
@click.option("--verbose", is_flag=True)
def cli(imgfile: str, pth_file: str, output: str, verbose: bool):
    evaluator = Evaluator(pth_file=pth_file)
    out = evaluator.evaluate(cv2.imread(imgfile), verbose)
    cv2.imwrite(output, out)


if __name__ == "__main__":
    cli()
