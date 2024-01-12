from functools import partial
from typing import Tuple

import lightning as L
import normflows as nf
import torch as T
import torch.nn as nn

from mltools.flows import rqs_flow

class InferenceFlow(L.LightningModule):
    network: nn.Module
    optimizer_factory: partial

    def __init__(self, data_sample: Tuple[T.Tensor, T.Tensor], conv_blocks_1: int,
            conv_blocks_2: int, hidden_conv_channels: int, mlp_depth: int,
            optimizer_factory: partial):
        super().__init__()

        self.save_hyperparameters(ignore=['data_sample', 'optimizer_factory'])

        conv_block = T.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
        )

        assert self.data_sample.shape[-1] == self.data_sample.shape[-2]
        img_size = self.data_sample.shape[-1]

        img_size_after_conv = (img_size - 2 * conv_blocks_1) / 2 + 2 * conv_blocks_2
        mlp_width = img_size_after_conv**2 / hidden_conv_channels
        assert img_size_after_conv**2 > 64
        self.network = T.Sequential(
            *[T.Sequential(nn.Conv2d(1, hidden_conv_channels, 3), nn.ReLU())
                for _ in range(conv_counts_1)],
            nn.MaxPool2d(2),
            *[T.Sequential(nn.Conv2d(1, hidden_conv_channels, 3), nn.ReLU())
                for _ in range(conv_counts_2)],
            nn.Flatten(),
            *[T.Sequential(nn.Linear(mlp_width, mlp_width), nn.ReLU())
                for _ in range(mlp_depth)],
            nn.Linear(mlp_width, 10),
            nn.Sigmoid(),
        )

        self.optimizer_factory = optimizer_factory

    def configure_optimizers(self) -> T.optim.Optimizer:
        return self.optimizer_factory(self.network.parameters())

    def training_step(self, batch : Tuple[T.Tensor, T.Tensor],
            batch_idx: int) -> T.Tensor:
        loss = self.network.forward_kld(*batch)

        self.log('train_loss', loss, on_step=True,
                on_epoch=True, logger=True)
        return dict(loss=loss)

    def validation_step(self, batch: Tuple[T.Tensor, T.Tensor],
            batch_idx: int) -> None:
        loss = self.network.forward_kld(*batch)

        self.log('valid_loss', loss, on_step=True,
                on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.network.forward_kld(*batch)

        self.log('test_loss', loss, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.network.forward(*batch)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    import unittest

    from ..datamodules.simple import InferenceDataModule
    from ..datamodules.preprocs import SimplePreprocessor
    from ..datamodules.datasets import MissingInfoToyDataset

    class TestInferenceFlow(unittest.TestCase):
        def test_training(self):
            preproc = SimplePreprocessor()
            train_set = MissingInfoToyDataset('prior', 10_000)
            train_set.acquire()

            loader_factory = partial(T.utils.data.DataLoader,
                    batch_size=1024, num_workers=1, pin_memory=True)
            datamod = InferenceDataModule(preproc, train_set, None,
                    loader_factory, None, test_frac=0.1, val_frac=0.1)
            datamod_pred = InferenceDataModule(preproc, train_set, 'use_test',
                    None, loader_factory, test_frac=0.1, val_frac=0.1)

            params = dict(
                num_blocks=5,
                mlp_width=64,
                mlp_depth=4,
                bin_count=10,
                spline_bound=10,
                optimizer_factory = partial(T.optim.Adam, lr=1.0e-3),
            )
            model = InferenceFlow(datamod.get_data_sample(), **params)

            trainer = L.Trainer(max_epochs=10, enable_checkpointing=False,
                    enable_progress_bar=True, enable_model_summary=False)
            trainer.fit(model, datamod)

            x, _ = train_set.get()

            batches = trainer.predict(model, datamod_pred)
            x_pred = datamod_pred.invert_setup_on_prediction(batches)

            fig, axs = plt.subplots(1, 2)
            axs[0].hist(x, density=True, histtype='step',
                    bins=40, color='C0', linestyle='dashed')
            axs[0].hist(x_pred, density=True, histtype='step',
                    bins=40, color='C0', linestyle='solid')
            plt.show()

        #def test_model(self):
        #    z, y = T.randn(1000, 1), T.randn(1000, 1)

        #    params = dict(
        #        num_blocks=16,
        #        mlp_width=256,
        #        mlp_depth=5,
        #        bin_count=10,
        #        spline_bound=5,
        #        optimizer_factory = partial(T.optim.Adam, lr=1.0e-5),
        #    )
        #    model = InferenceFlow((z[:1], y[:1]), **params)
        #    x = model.network.inverse(z, y).detach().numpy().reshape(-1)
        #    z = z.reshape(-1)
        #    y = y.reshape(-1)

        #    fig, ax = plt.subplots(1, 1)

        #    ax.hist(z, density=True, histtype='step',
        #            bins=40, color='C0', linestyle='dashed')
        #    ax.hist(x, density=True, histtype='step',
        #            bins=40, color='C0')

        #    plt.show()
            

    unittest.main()
