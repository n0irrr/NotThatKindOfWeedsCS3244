"""Pre-trained backbones (e.g. ResNet50, MobileNetV2) for transfer learning
and fine-tuning. Reference: DeepWeeds paper reports 95.7% with ResNet50."""


def build(backbone: str, num_classes: int, freeze_backbone: bool = True):
    raise NotImplementedError
