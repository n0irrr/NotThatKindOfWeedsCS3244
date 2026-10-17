"""Small CNN trained from scratch."""
import torch.nn as nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def build(num_classes: int, params: dict):
    """Conv blocks (2x conv-BN-ReLU + maxpool each) -> global avg pool ->
    dropout -> linear. params: channels (list), dropout."""
    channels = params.get("channels", [32, 64, 128, 256])
    blocks = [_block(c_in, c_out) for c_in, c_out in zip([3] + channels[:-1], channels)]
    return nn.Sequential(
        *blocks,
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(params.get("dropout", 0.3)),
        nn.Linear(channels[-1], num_classes),
    )
