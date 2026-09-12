"""Hand-crafted feature representations for classical ML baselines.

Per the proposal: flattened pixels, downsampled pixels, colour histograms.
"""


def flattened_pixels(images, size=(32, 32)):
    raise NotImplementedError


def colour_histogram(images, bins=32):
    raise NotImplementedError
