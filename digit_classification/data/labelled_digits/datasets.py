import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from torchvision import datasets
from torchvision import transforms

class LabelledDigitsDataset(ABC):
    ...

class MNISTDataset(LabelledDigitsDataset):
    labels: np.ndarray
    digit_imgs: np.ndarray

    def __init__(
        self,
        *,
        load_path: str,
        size: Optional[int] = None, # we might want to restrict size for debugging
        train: bool,
    ) -> None:
        super().__init__()

        mnist = datasets.MNIST(load_path, train=train, download=True)

        self.labels = mnist.targets
        self.digit_imgs = mnist.data
        if size:
            assert size <= len(mnist.targets)
            self.labels = self.labels[:size]
            self.digit_imgs = self.digit_imgs[:size]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx_like: int | slice) -> Tuple[NDArray, NDArray]:
        return self.labels[idx_like], self.digit_imgs[idx_like]
