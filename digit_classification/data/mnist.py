from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, Dict, Iterable, Literal, Optional, Tuple, Union

import hydra
import lightning as L
import numpy as np
import torch as T
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset, TensorDataset
from mltools.torch_utils import train_valid_split
from omegaconf import DictConfig
from torchvision.datasets import MNIST

class MNISTDataset:
    labels: np.ndarray
    digit_imgs: np.ndarray

    def __init__(
        self,
        *,
        load_path: str,
        size: Optional[int] = None, # might restrict size for debugging
        train: bool,
    ) -> None:
        super().__init__()

        mnist = MNIST(load_path, train=train, download=True)

        # want numpy here, because we want to use this class during plotting
        self.labels = mnist.targets.numpy()
        self.digit_imgs = mnist.data.numpy()
        if size:
            assert size <= len(mnist.targets)
            self.labels = self.labels[:size]
            self.digit_imgs = self.digit_imgs[:size]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx_like: int | slice) -> Tuple[NDArray, NDArray]:
        return self.labels[idx_like], self.digit_imgs[idx_like]

class StandardDataModule(L.LightningDataModule):
    hparams: Any
    train_set: Dataset
    valid_set: Dataset
    test_set: Dataset
    predict_set: Dataset

    def __init__(
        self,
        *,
        dev_set_conf: DictConfig,
        predict_set_conf: DictConfig,
        val_frac: float = 0.1,
        dev_loader_conf: DictConfig,
        predict_loader_conf: DictConfig,
        crop_at_size: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage in ["fit", "val"] and not hasattr(self, "train_set"):
            dev_set = hydra.utils.instantiate(self.hparams.dev_set_conf)
            self.train_set, self.val_set = train_valid_split(
                dev_set, self.hparams.val_frac, split_type="rand"
            )
        elif stage in ["fit", "val"] and hasattr(self, "train_set"): 
            assert hasattr(self, "val_set")
        elif stage == "test":
            pass # no idea for what testing would be useful
        elif stage == "predict":
            self.predict_set = hydra.utils.instantiate(self.hparams.predict_set_conf)
        else:
            assert False

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, shuffle=True, **self.hparams.dev_loader_conf)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, shuffle=True, **self.hparams.dev_loader_conf)

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_set,
            shuffle=False,
            drop_last=False,
            **self.hparams.predict_loader_conf,
        )

    def mock_sample(self) -> tuple[T.Tensor, T.Tensor]:
        return T.zeros((1,)), T.zeros((1, 28, 28))

    def invert_setup_on_prediction(self, batches: Iterable) -> dict[str, T.Tensor]:
        return self._unpreproc(T.cat(batches))

    def _preproc(self, dataset: MNISTDataset) -> Dataset:
        labels, digit_imgs = dataset[:]
        return TensorDataset(torch.tensor(labels), torch.tensor(digit_imgs))

    def _unpreproc(self, labels: T.Tensor) -> NDArray:
        return labels.numpy()


if __name__ == "__main__":
    import tempfile
    import unittest

    from omegaconf import OmegaConf
    
    class TestLabelledDigitsModule(unittest.TestCase):
        def test_setup(self):
            with tempfile.TemporaryDirectory() as load_path:
                dev_set_conf = OmegaConf.create(dict(
                    _target_="digit_classification.data.mnist.MNISTDataset",
                    load_path=load_path,
                    train=True,
                    size=1,
                ))
                predict_set_conf = OmegaConf.create(dict(
                    _target_="digit_classification.data.mnist.MNISTDataset",
                    load_path=load_path,
                    train=False,
                    size=1,
                ))

                dev_loader_conf = OmegaConf.create(dict(batch_size=1))
                predict_loader_conf = OmegaConf.create(dict(batch_size=1))

                labels_pre = hydra.utils.instantiate(dev_set_conf)[:][0]

                datamod = StandardDataModule(
                    dev_set_conf=dev_set_conf,
                    predict_set_conf=predict_set_conf,
                    val_frac=0.1,
                    dev_loader_conf=dev_loader_conf,
                    predict_loader_conf=predict_loader_conf
                )
                datamod.setup("fit")
                datamod.setup("val")
                datamod.setup("predict")

                dataloader = datamod.train_dataloader()

                for _labels, _digit_imgs in dataloader:
                    print("digit_imgs.shape:", _digit_imgs.shape)
                    print("labels.shape:", _labels.shape)
                    labels_post = datamod.invert_setup_on_prediction([_labels])
                    break

            self.assertTrue(len(labels_pre) == 1)
            self.assertTrue(len(labels_post) == 1)
            self.assertTrue(labels_pre == labels_post)

        def test_mock_sample(self):
            with tempfile.TemporaryDirectory() as load_path:
                dev_set_conf = OmegaConf.create(dict(
                    _target_="digit_classification.data.mnist.MNISTDataset",
                    load_path=load_path,
                    size=1,
                    train=True
                ))
                predict_set_conf = OmegaConf.create(dict(
                    _target_="digit_classification.data.mnist.MNISTDataset",
                    load_path=load_path,
                    size=1,
                    train=False
                ))

                dev_loader_conf = OmegaConf.create(dict(batch_size=1))
                predict_loader_conf = OmegaConf.create(dict(batch_size=1))

                datamod = StandardDataModule(
                    dev_set_conf=dev_set_conf,
                    predict_set_conf=predict_set_conf,
                    val_frac=0.1,
                    dev_loader_conf=dev_loader_conf,
                    predict_loader_conf=predict_loader_conf
                )
                datamod.setup("fit")
                
                for labels, digit_imgs in datamod.train_dataloader():
                    sample_real = labels, digit_imgs
                mock_sample = datamod.mock_sample()
                self.assertListEqual(
                    [sample_real[i].shape for i in range(len(sample_real))],
                    [mock_sample[i].shape for i in range(len(sample_real))],
                )

    unittest.main()
