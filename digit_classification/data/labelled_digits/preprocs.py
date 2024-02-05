from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
import torch as T
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .datasets import LabelledDigitsDataset
from ..datamod import StandardPreprocessor


class LabelledDigitsPreprocessor(StandardPreprocessor):
    labels: T.tensor
    digit_imgs: T.tensor

    class PostProcDataset(StandardPreprocessor.PostProcDataset):
        def __init__(self, labels: T.Tensor, digit_imgs: T.Tensor) -> None:
            self.labels = labels
            self.digit_imgs = digit_imgs

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx_like: int | slice) -> tuple[T.Tensor, T.Tensor]:
            return self.labels[idx_like], self.digit_imgs[idx_like]

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def preproc(self, dataset: LabelledDigitsDataset) -> Any:
        ...

    @abstractmethod
    def unpreproc(self, label_batches: Iterable[T.Tensor]) -> tuple[NDArray]:
        ...

    def mock_sample(self) -> tuple[T.Tensor, T.Tensor]:
        return T.zeros((1,)), T.zeros((1, 1, 28, 28))


class SimplePreprocessor(LabelledDigitsPreprocessor):
    def __init__(self) -> None:
        super().__init__()

    def preproc(self, dataset: LabelledDigitsDataset) -> Any:
        labels, digit_imgs = dataset[:]
        labels = self._preproc_labels(labels)
        digit_imgs = self._preproc_digit_imgs(digit_imgs)
        return self.PostProcDataset(labels, digit_imgs)


    def unpreproc(self, label_batches: Iterable[T.Tensor]) -> tuple[NDArray]:
        labels = T.cat(label_batches)
        return (self._unpreproc_labels(labels),)

    def _preproc_labels(self, labels: NDArray) -> T.Tensor:
        return T.tensor(labels)

    def _preproc_digit_imgs(self, digit_imgs: NDArray) -> T.Tensor:
        assert digit_imgs.shape[-1] == digit_imgs.shape[-2]
        img_size = digit_imgs.shape[-1]
        digit_imgs = digit_imgs.reshape(len(digit_imgs), 1, img_size, img_size)
        return T.tensor(digit_imgs)

    def _unpreproc_labels(self, labels: T.Tensor) -> NDArray:
        return labels.numpy()


if __name__ == "__main__":
    import tempfile
    import unittest

    from .datasets import MNISTDataset

    class TestSimplePreprocessor(unittest.TestCase):
        def test_mock_sample(self):
            preproc = SimplePreprocessor()
            with tempfile.TemporaryDirectory() as load_path:
                dataset = MNISTDataset(load_path=load_path, size=1)

            sample_real = preproc.preproc(dataset)[:]
            sample_mock = preproc.mock_sample()
            self.assertListEqual(
                [sample_real[i].shape for i in range(len(sample_real))],
                [sample_mock[i].shape for i in range(len(sample_real))],
            )

        def test_consistency(self):
            preproc = SimplePreprocessor()

            labels = np.random.rand(10, 10)
            labels_preproc = preproc._preproc_labels(labels)
            labels_reco = preproc._unpreproc_labels(labels_preproc)

            reco_err = np.abs(labels - labels_reco)
            self.assertTrue(np.mean(reco_err) < 1e-6)

            print("reco_err_mean:", np.mean(reco_err))
            print("reco_err_std:", np.std(reco_err))

    unittest.main()
