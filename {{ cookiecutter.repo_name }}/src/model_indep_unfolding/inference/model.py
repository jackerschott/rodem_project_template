import hydra
import pytorch_lightning as pl
import torch as T
import torch.nn as nn
from typing import Any, Tuple, Dict

from ..cfg import get_config

class InferenceFlow(pl.LightningModule):
    network: nn.Module
    optimizer_cfg: Dict[str, Any]

    def __init__(self, network: nn.Module, lr: float):
        super().__init__()
        self.network = network
        self.optimizer_cfg = dict(lr=lr)

    def configure_optimizers(self) -> T.optim.Optimizer:
        cfg = get_config()
        return T.optim.Adam(self.network.parameters(), **self.optimizer_cfg)

    def training_step(self, batch : Tuple[T.Tensor, T.Tensor],
            batch_idx: int) -> T.Tensor:
        x, c = batch

        z, log_jac_det = self.network.forward(x, c)
        loss = train_loss(z, log_jac_det)

        self.log('train_loss', loss, on_step=False,
                on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[T.Tensor, T.Tensor],
            batch_idx: int) -> None:
        x, c = batch
        
        z, log_jac_det = self.forward(x, c)
        loss = train_loss(z, log_jac_det)

        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        z, c = batch

        x, log_jac_det = self.forward(z, c, rev=True)
        return x, log_jac_det

    def forward(self, x: T.Tensor, c: T.Tensor, rev: bool = False,
            jac: bool = True) -> Tuple[T.Tensor, T.Tensor]:
        return self.network(x, c, rev, jac)

def train_loss(z: T.Tensor, log_jac_det: T.Tensor) -> T.Tensor:
    # z: (batch, feature), log_jac_det: (batch,)
    #return T.mean(T.sum(0.5 * z**2, axis=-1) - log_jac_det)
    return -T.mean(log_jac_det)
