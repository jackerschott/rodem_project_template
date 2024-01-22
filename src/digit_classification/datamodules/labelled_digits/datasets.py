import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from torchvision import datasets


class Dataset(ABC):
    labels: NDArray
    digit_imgs: NDArray

    def __init__(
        self, size: Optional[int] = None, load_path: Optional[str] = None
    ) -> None:
        if load_path is not None:
            self.load(load_path)
            self.size = len(self.labels)
        elif size is not None:
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
    def __init__(
        self, size: Optional[int] = None, load_path: Optional[str] = None
    ) -> None:
        super().__init__(size, load_path)

    def acquire(self, tmp_save_dir=None) -> None:
        # tmp_save_dir is useful for testing to avoid re-downloading the dataset
        # every time; in a real workflow the dataset should be cached manually with
        # the save and load methods
        if tmp_save_dir is None:
            with tempfile.TemporaryDirectory() as root_path:
                train_set = datasets.MNIST(root_path, train=True, download=True)
                test_set = datasets.MNIST(root_path, train=False, download=True)
        else:
            train_set = datasets.MNIST(tmp_save_dir, train=True, download=True)
            test_set = datasets.MNIST(tmp_save_dir, train=False, download=True)

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
        return labels[:1], digit_imgs[:1]

    def save(self, filename: str) -> None:
        np.savez_compressed(filename, self.labels, self.digit_imgs)

    def load(self, filename: str) -> None:
        self.labels, self.digit_imgs = np.load(filename).values()
