import torch


class Config:
    epoch = 100
    learning_rate = 1e-3
    batch_size = 4
    device = torch.device("cpu")


# change it on argparse
config = Config()
