"""Pre-trained backbones (e.g. ResNet50, MobileNetV2) for transfer learning
and fine-tuning. Reference: DeepWeeds paper reports 95.7% with ResNet50."""
import torch.nn as nn
from torchvision import models


def build(backbone: str, num_classes: int, freeze_backbone: bool = True,
          dropout: float = 0.3, pretrained: bool = True):
    """ImageNet backbone with a new Dropout -> Linear head as `model.fc`."""
    if backbone != "resnet50":
        raise ValueError(f"unsupported backbone {backbone!r}")
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    set_backbone_trainable(model, not freeze_backbone)
    return model


def set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze/unfreeze every parameter except the classification head."""
    for name, p in model.named_parameters():
        p.requires_grad = trainable or name.startswith("fc.")
