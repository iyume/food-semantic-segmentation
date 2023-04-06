from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class GeneratedDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, *, dataset_dir: str = "dataset") -> None:
        self.image_dir = Path(dataset_dir) / "images"
        self.label_dir = Path(dataset_dir) / "labels"
        self.image_files = list(self.image_dir.iterdir())
        self.label_files = list(self.label_dir.iterdir())
        assert len(self.image_files) >= 40, "dataset is not enough"
        # auto generate test
        self.num_test = len(self.image_files) // 5

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = cv2.imread(str(self.image_files[index]))
        label = cv2.imread(str(self.label_files[index]), cv2.IMREAD_GRAYSCALE)
        return self.transform(image), torch.from_numpy(label).to(torch.long)

    @staticmethod
    def transform(im: np.ndarray) -> Tensor:
        return ToTensor()(im)

    def get_testdata(self) -> Tuple[Tensor, Tensor]:
        """Get last 20% dataset for test."""
        images = []
        labels = []
        for i in range(len(self.image_files) - self.num_test, len(self.image_files)):
            image, label = self[i]
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.stack(labels)

    def __len__(self) -> int:
        return len(self.image_files) - self.num_test
