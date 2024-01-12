from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import tempfile
from typing import Any, Tuple, Dict, Optional

from torchvision import datasets

class Dataset(ABC):
    labels: NDArray
    digit_imgs: NDArray

    def __init__(self, size: Optional[int] = None,
            load_path: Optional[str] = None) -> None:
        if size is None and load_path is not None:
            self.load(load_path)
            self.size = len(self.labels)
        elif size is not None and load_path is None:
            self.size = size
        else:
            assert False

    @abstractmethod
    def acquire(self) -> None:
        ...

    @abstractmethod
    def get(self) -> Tuple[NDArray, NDArray]:
        ...

    @abstractmethod
    def get_init_sample(self) -> Tuple[NDArray, NDArray]:
        ...

    @abstractmethod
    def save(self, filename: str) -> None:
        ...

    @abstractmethod
    def load(self, filename: str) -> None:
        ...

class MNISTDataset(Dataset):
    def __init__(self, size: Optional[int] = None,
            load_path: Optional[str] = None) -> None:
        super().__init__(size, load_path)

    def acquire(self) -> None:
        with tempfile.TemporaryDirectory() as root_path:
            train_set = datasets.MNIST(root_path, train=True, download=True)
            test_set = datasets.MNIST(root_path, train=False, download=True)

        assert self.size <= len(train_set.targets) + len(test_set.targets)
        perm = np.random.permutation(self.size)
        self.labels = np.concatenate((train_set.targets, test_set.targets))[perm]
        self.digit_imgs = np.concatenate((train_set.data, test_set.data))[perm]

        # this is not preprocessing, I just
        # don't care for channels between 0 and 255
        self.digit_imgs = self.digit_imgs / 255.0

    def get(self) -> Tuple[NDArray, NDArray]:
        return self.labels, self.digit_imgs

    def get_init_sample(self) -> Tuple[NDArray, NDArray]:
        labels, digit_imgs = self.get()
        return labels[0], digit_imgs[0]

    def save(self, filename: str) -> None:
        np.savez_compressed(filename, self.labels, self.digit_imgs)

    def load(self, filename: str) -> None:
        self.labels, self.digit_imgs = np.load(filename).values()
