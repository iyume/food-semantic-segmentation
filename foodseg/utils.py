from typing import Any, Dict, TypedDict

import torch


class State(TypedDict):
    """The training state."""

    epoch: int
    model_state_dict: Dict[str, torch.Tensor]
    optim_state_dict: Dict[str, Any]
    loss: torch.Tensor


id2color = {
    0: (0, 0, 0),
    1: (150, 0, 250),
    2: (200, 50, 250),
    3: (30, 100, 250),
    4: (30, 0, 200),
    5: (150, 50, 200),
    6: (200, 100, 200),
    7: (30, 0, 150),
    8: (30, 50, 150),
    9: (150, 100, 150),
    10: (200, 0, 100),
    11: (30, 50, 100),
    12: (255, 255, 255),
}  # id: (R, G, B)
classes = {
    "背景": 0,
    "白切鸡": 1,
    "炒西兰花": 2,
    "干煸豆子": 3,
    "红烧茄子": 4,
    "红烧鱼": 5,
    "家常豆腐": 6,
    "尖椒肉丝": 7,
    "凉拌黄瓜": 8,
    "土豆牛肉": 9,
    "土豆丝": 10,
    "西红柿鸡蛋": 11,
    "盘子": 12,
}
