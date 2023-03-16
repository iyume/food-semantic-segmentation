import torch
from torch import nn
from torchvision.models.segmentation import DeepLabV3, deeplabv3_resnet50


def get_model(num_classes: int) -> DeepLabV3:
    # model = deeplabv3_resnet50(num_classes=num_classes)
    model = deeplabv3_resnet50(num_classes=num_classes, pretrained_backbone=False)
    missing_keys, unexpected_keys = model.backbone.load_state_dict(
        torch.load("./pretrained/resnet50-0676ba61.pth"), strict=False
    )
    # "fc.weight" and "fc.bias" is dropped
    if missing_keys:
        raise RuntimeError
    return model


class Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 50, 3, 1, 1)
        self.aux = nn.Conv2d(50, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.aux(x)
        return x
