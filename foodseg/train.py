from pathlib import Path

import click
import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from .datasets import GeneratedDataset
from .models.unet import UNet
from .utils import State

device = torch.device("cpu")
learning_rate = 1e-3
# epoch increase in training
num_epochs = 100
batch_size = 4

dataset = GeneratedDataset(dataset_dir="dataset-compressed")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = UNet(3, 12)
# load temp model
model.load_state_dict(torch.load("", device))

model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
loss_fn = nn.CrossEntropyLoss()

ckpt_dir = Path(".ckpt")
ckpt_dir.mkdir(exist_ok=True)


def train(state: State, create_checkpoint: bool = True):
    checkpoint = ckpt_dir / f"model_epoch{state['epoch']}.pth"
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        print("iter: {}  loss: {:.4f}".format(idx, loss))
    print(f"epoch {state['epoch']} training complete")
    state["epoch"] += 1
    if create_checkpoint:
        torch.save(model.state_dict(), checkpoint)
        print(f"model saved at {checkpoint}")


@click.command()
@click.option()
def cli():
    ...


def test():
    ...


if __name__ == "__main__":
    test()
