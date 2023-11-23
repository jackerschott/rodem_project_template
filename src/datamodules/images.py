from copy import deepcopy
from functools import partial
from typing import Mapping

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_set: partial,
        test_set: partial,
        loader_config: Mapping,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Load the validation set now to get a single sample (to initialise network)
        self.test_set = test_set()  # For now test=valid

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""

        if stage in ["train", "fit"]:
            self.train_set = self.hparams.train_set()
        if stage in ["test", "predict"]:
            self.valid_set = self.hparams.valid_set()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams.loader_config)

    def test_dataloader(self) -> DataLoader:
        test_config = deepcopy(self.hparams.loader_config)
        test_config["drop_last"] = False
        test_config["shuffle"] = False
        return DataLoader(self.test_set, **test_config)

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_data_sample(self) -> tuple:
        """Get a single data sample to help initialise the neural network."""
        return self.test_set[0]
