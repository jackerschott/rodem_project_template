from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, Dict, Iterable, Literal, Optional, Tuple, Union

import hydra
import lightning as L
import numpy as np
import torch as T
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from mltools.torch_utils import train_valid_split
from omegaconf import DictConfig

# return a torch dataset in preprocessor instead of making the dataset we pass to it a
# torch dataset; this is so we can keep any ML stuff out of the dataset class we pass in
# load and keep it as simple as possible; this is generally helpful but also helpful in
# practice, since we want to use it when plotting truth vs prediction, where no ML stuff
# or preprocessed data is needed; in the cases where it is needed, one can still write a
# plotting script which uses preprocessor classes or the whole datamodule
class StandardPreprocessor(ABC):
    class PostProcDataset(ABC, Dataset):
        @abstractmethod
        def __len__(self) -> int:
            ...

        @abstractmethod
        def __getitem__(self, idx_like: int | slice) -> Any:
            ...

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preproc(self, dataset: Any) -> PostProcDataset:
        ...

    @abstractmethod
    def unpreproc(self, batches: Iterable) -> Any:
        ...

    @abstractmethod
    def mock_sample(self):
        ...


class StandardDataModule(L.LightningDataModule):
    hparams: Any
    preproc: StandardPreprocessor
    train_set: Dataset
    valid_set: Dataset
    test_set: Dataset
    predict_set: Dataset

    def __init__(
        self,
        *,
        dev_set_conf: DictConfig,
        predict_set_conf: DictConfig,
        preproc_conf: DictConfig,
        val_frac: float = 0.1,
        dev_loader_conf: DictConfig,
        predict_loader_conf: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.preproc = hydra.utils.instantiate(self.hparams.preproc_conf)

    def setup(self, stage: str):
        if stage in ["fit", "val"] and not hasattr(self, "train_set"):
            dev_set = self.preproc.preproc(
                hydra.utils.instantiate(self.hparams.dev_set_conf)
            )
            self.train_set, self.val_set = train_valid_split(
                dev_set, self.hparams.val_frac, split_type="rand"
            )
        elif stage in ["fit", "val"] and hasattr(self, "train_set"): 
            assert hasattr(self, "val_set")
        elif stage == "test":
            pass # no idea for what testing would be useful
        elif stage == "predict":
            self.predict_set = self.preproc.preproc(
                hydra.utils.instantiate(self.hparams.predict_set_conf)
            )
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

    def mock_sample(self) -> Any:
        return self.preproc.mock_sample()

    def invert_setup_on_prediction(
        self, batches: Iterable
    ) -> dict[str, T.Tensor]:
        return self.preproc.unload(batches)


if __name__ == "__main__":
    import tempfile
    import unittest

    from omegaconf import OmegaConf
    
    from .labelled_digits.datasets import MNISTDataset
    from .labelled_digits.preprocs import SimplePreprocessor

    class TestLabelledDigitsModule(unittest.TestCase):
        def test_setup(self):
            with tempfile.TemporaryDirectory() as load_path:
                dev_set_conf = OmegaConf.create(dict(
                    _target_="digit_classification.data.labelled_digits.datasets.MNISTDataset",
                    load_path=load_path,
                    size=1,
                    train=True
                ))
                predict_set_conf = OmegaConf.create(dict(
                    _target_="digit_classification.data.labelled_digits.datasets.MNISTDataset",
                    load_path=load_path,
                    size=1,
                    train=False
                ))
                preproc_conf = OmegaConf.create(dict(
                    _target_="digit_classification.data.labelled_digits.preprocs.SimplePreprocessor",
                ))

                dev_loader_conf = OmegaConf.create(dict(batch_size=1))
                predict_loader_conf = OmegaConf.create(dict(batch_size=1))

                datamod = StandardDataModule(
                    dev_set_conf=dev_set_conf,
                    predict_set_conf=predict_set_conf,
                    preproc_conf=preproc_conf,
                    val_frac=0.1,
                    dev_loader_conf=dev_loader_conf,
                    predict_loader_conf=predict_loader_conf
                )
                datamod.setup("fit")
                datamod.setup("val")
                datamod.setup("predict")

            dataloader = datamod.train_dataloader()

            for _labels, _digit_imgs in dataloader:
                breakpoint()
                print("digit_imgs.shape:", _digit_imgs.shape)
                print("labels.shape:", _labels.shape)
                labels = dataloader.invert_setup_on_prediction([_labels])
                break

            dev_set = hydra.utils.instantiate(dev_set_conf)
            self.assertTrue(labels.shape == dev_set[0][0])

        #def test_get_data_sample(self):
        #    preproc = SimplePreprocessor()
        #    train_set = MNISTDataset(10_000)
        #    train_set.acquire(tmp_save_dir="mnist")

        #    train_loader_factory = partial(
        #        T.utils.data.DataLoader, batch_size=1024, num_workers=1, pin_memory=True
        #    )
        #    datamod = LabelledDigitsModule(
        #        preproc,
        #        train_set,
        #        None,
        #        train_loader_factory,
        #        None,
        #        test_frac=0.1,
        #        val_frac=0.1,
        #    )

        #    label, digit_img = datamod.get_data_sample()
        #    self.assertTrue(label.shape == (1, 10))
        #    self.assertTrue(T.sum(label == 0.0) == 9 and T.sum(label == 1.0) == 1)

        #    self.assertTrue(digit_img.shape == (1, 1, 28, 28))
        #    self.assertTrue(T.all((digit_img >= 0.0) & (digit_img <= 1.0)))
        #    self.assertTrue(T.all((digit_img >= 0.0) & (digit_img <= 1.0)))

    unittest.main()
