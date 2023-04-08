from pathlib import Path
from typing import Optional, cast

import click
import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from .datasets import GeneratedDataset
from .models.unet import UNet
from .utils import State, classes

device = torch.device("cpu")
learning_rate = 1e-3
# epoch increase in training
num_epochs = 100
batch_size = 4

dataset = GeneratedDataset(dataset_dir="dataset-compressed")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = UNet(3, len(classes))
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
loss_fn = nn.CrossEntropyLoss()

ckpt_dir = Path(".ckpt")
ckpt_dir.mkdir(exist_ok=True)


def train_one(state: State, create_checkpoint: bool = True):
    loss_history = []
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss: torch.Tensor = loss_fn(out, labels)
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
        print("iter: {}  loss: {:.4f}".format(idx, loss))
    state["epoch"] += 1
    state["loss"] = sum(loss_history) / len(loss_history)
    print(f"epoch {state['epoch']} training complete")
    if create_checkpoint:
        checkpoint = ckpt_dir / f"model_epoch{state['epoch']}.pth"
        torch.save(state, checkpoint)
        print(f"model saved at {checkpoint}")


def train(pth: Optional[str] = None):
    if pth is None:
        state = State(
            epoch=0,
            model_state_dict=model.state_dict(),
            optim_state_dict=optimizer.state_dict(),
            loss=0,
        )
    else:
        state = cast(State, torch.load(pth, device))
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optim_state_dict"])
    for _ in range(num_epochs):
        train_one(state)


@click.command()
@click.option(
    "-p", "--pth-file", default=None, type=click.Path(exists=True, dir_okay=False)
)
def cli(pth_file: Optional[str]):
    train(pth_file)


if __name__ == "__main__":
    cli()
