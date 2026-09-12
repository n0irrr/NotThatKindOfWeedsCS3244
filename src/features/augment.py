"""Data augmentation: rotate, flip, crop/zoom, brightness/contrast/colour,
translation, noise/blur."""


def get_train_transforms(config):
    raise NotImplementedError


def get_eval_transforms(config):
    raise NotImplementedError
