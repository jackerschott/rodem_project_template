from abc import ABC, abstractmethod
import hydra
import multiprocessing
from numpy.typing import NDArray
import pytorch_lightning as pl
import torch as T
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from typing import Any, Optional, Dict
import wandb

from ..datasets import Dataset
from .preprocs import Preprocessor, SimplePreprocessor
from .model import InferenceFlow
from ..cfg import get_config

# flow testing
from mltools.mltools.flows import rqs_flow

class Inferer:
    model: pl.LightningModule
    datamod: pl.LightningDataModule
    preproc: Preprocessor
    trainer: pl.Trainer

    def __init__(self, data_dim: int) -> None:
        cfg = get_config()

        self.preproc = hydra.utils.instantiate(
                cfg.inferer.preproc, data_dim=data_dim)
        #network = hydra.utils.instantiate(cfg.inferer.network,
        #        inn_dim_in=self.preproc.post_dim('hidden'),
        #        cond_dim_in=self.preproc.post_dim('visible'))
        network = rqs_flow(self.preproc.post_dim('hidden'),
                self.preproc.post_dim('visible'), mlp_width=256, mlp_depth=5,
                mlp_act=nn.ReLU, num_bins=10, do_lu=False)
        
        self.model = InferenceFlow(network, lr=cfg.inferer.model.lr)

    def set_data(self, x_train: NDArray, y_train: NDArray,
            x_predict: Optional[NDArray] = None,
            y_predict: Optional[NDArray] = None) -> None:
        # set data before fit/predict: if both datasets are the same, we need to
        # determine splits
        cfg = get_config()
        self.datamod = hydra.utils.instantiate(cfg.inferer.datamod,
                x_train, y_train, x_predict, y_predict, self.preproc)

    def fit(self) -> None:
        cfg = get_config()
        self.trainer: pl.Trainer = hydra.utils.instantiate(cfg.inferer.trainer,
                enable_progress_bar=True, log_every_n_steps=1)
        self.trainer.fit(self.model, self.datamod)

    def predict(self) -> NDArray:
        batches = self.trainer.predict(self.model, datamodule=self.datamod)
        x = T.cat([batch for batch in batches])
        # TODO: use this by converting it to density value
        #log_jac_det = torch.stack([batch[1] for batch in batches])
        return x.numpy()
