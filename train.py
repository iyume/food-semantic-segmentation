from pathlib import Path

import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from backend.models.unet import UNet
from config import config
from datasets import GeneratedDataset

model = UNet(3, 12)
epoch_start = 1

# load temp model
model.load_state_dict(torch.load("pretrained/model_ep42.pth", config.device))
epoch_start = 43

model.to(config.device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
loss_fn = nn.CrossEntropyLoss()

dataset = GeneratedDataset(dataset_dir="dataset-compressed")
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

ckpt_dir = Path(".ckpt")
ckpt_dir.mkdir(exist_ok=True)

for epoch in range(epoch_start, epoch_start + config.epoch):
    checkpoint = ckpt_dir / f"model_ep{epoch}.pth"
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(config.device)
        labels = labels.to(config.device)
        optimizer.zero_grad()
        out = model(images)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        print("iter: {}  loss: {:.4f}".format(idx, loss))
    print(f"epoch {epoch} training complete")
    torch.save(model.state_dict(), checkpoint)
    print(f"model saved at {checkpoint}")
