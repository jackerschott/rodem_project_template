from functools import partial
from typing import Tuple

import lightning as L
import numpy as np
import torch as T
import torch.nn as nn


class ConvClassifier(L.LightningModule):
    IMG_SIZE = 28

    network: nn.Module
    optimizer_factory: partial

    def __init__(
        self,
        data_sample: Tuple[T.Tensor, T.Tensor],
        conv_blocks_1: int,
        conv_blocks_2: int,
        hidden_conv_channels: int,
        mlp_depth: int,
        optimizer_factory: partial,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["data_sample", "optimizer_factory"])

        label_sample, digit_img_sample = data_sample

        assert (
            digit_img_sample.shape[-1] == self.IMG_SIZE
            and digit_img_sample.shape[-2] == self.IMG_SIZE
        )
        assert conv_blocks_1 > 0 and conv_blocks_2 > 0
        img_size_after_conv_1 = self.IMG_SIZE - 2 * conv_blocks_1
        img_size_after_conv_1 //= 2  # max pooling
        img_size_after_conv_2 = img_size_after_conv_1 - 2 * conv_blocks_2
        mlp_width = img_size_after_conv_2**2 * hidden_conv_channels

        self.network = nn.Sequential(
            nn.Conv2d(1, hidden_conv_channels, 3),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Conv2d(hidden_conv_channels, hidden_conv_channels, 3), nn.ReLU()
                )
                for _ in range(conv_blocks_1 - 1)
            ],
            nn.MaxPool2d(2),
            *[
                nn.Sequential(
                    nn.Conv2d(hidden_conv_channels, hidden_conv_channels, 3), nn.ReLU()
                )
                for _ in range(conv_blocks_2)
            ],
            nn.Flatten(),
            *[
                nn.Sequential(nn.Linear(mlp_width, mlp_width), nn.ReLU())
                for _ in range(mlp_depth)
            ],
            nn.Linear(mlp_width, label_sample.shape[-1]),
            nn.Softmax(dim=-1),
        )

        self.optimizer_factory = optimizer_factory

        self.loss = nn.BCELoss()

    def configure_optimizers(self) -> T.optim.Optimizer:
        return self.optimizer_factory(self.network.parameters())

    def training_step(
        self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int
    ) -> T.Tensor:
        labels, digit_imgs = batch
        labels_pred = self.network(digit_imgs)
        loss = self.loss(labels_pred, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return dict(loss=loss)

    def validation_step(self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int) -> None:
        labels, digit_imgs = batch
        labels_pred = self.network(digit_imgs)
        loss = self.loss(labels_pred, labels)

        self.log(
            "valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        labels, digit_imgs = batch
        labels_pred = self.network(digit_imgs)
        loss = self.loss(labels_pred, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        labels, digit_imgs = batch
        return self.network(digit_imgs), labels


if __name__ == "__main__":
    import multiprocessing
    import unittest

    from ...datamodules.labelled_digits.datamod import LabelledDigitsModule
    from ...datamodules.labelled_digits.datasets import MNISTDataset
    from ...datamodules.labelled_digits.preprocs import SimplePreprocessor

    class TestConvClassifier(unittest.TestCase):
        def test_network(self):
            digit_imgs = T.rand(10, 1, 28, 28)
            labels = T.rand(10, 10)

            model = ConvClassifier(
                (labels[:1], digit_imgs[:1]),
                conv_blocks_1=2,
                conv_blocks_2=2,
                hidden_conv_channels=8,
                mlp_depth=3,
                optimizer_factory=partial(T.optim.Adam, lr=1.0e-3),
            )
            labels_pred = model.network(digit_imgs).detach()
            self.assertTrue(labels_pred.shape == labels.shape)
            self.assertTrue(T.all((labels_pred >= 0) & (labels_pred <= 1)))

            sum_one_diff = T.abs(T.sum(labels_pred, dim=-1) - 1)
            print("sum_diff_mean =", T.mean(sum_one_diff))
            print("sum_diff_std =", T.std(sum_one_diff))
            self.assertTrue(T.mean(sum_one_diff) < 1e-4)

        def test_training(self):
            preproc = SimplePreprocessor()
            train_set = MNISTDataset(1_000)
            train_set.acquire(tmp_save_dir="mnist")

            train_loader_factory = partial(
                T.utils.data.DataLoader,
                batch_size=32,
                num_workers=(multiprocessing.cpu_count() - 1),
                pin_memory=True,
            )
            datamod = LabelledDigitsModule(
                preproc,
                train_set,
                "use_test",
                train_loader_factory,
                train_loader_factory,
                test_frac=0.1,
                val_frac=0.1,
            )

            model = ConvClassifier(
                datamod.get_data_sample(),
                conv_blocks_1=2,
                conv_blocks_2=2,
                hidden_conv_channels=8,
                mlp_depth=3,
                optimizer_factory=partial(T.optim.Adam, lr=1.0e-3),
            )

            trainer = L.Trainer(
                max_epochs=1,
                enable_checkpointing=False,
                enable_progress_bar=True,
                enable_model_summary=True,
            )
            trainer.fit(model, datamod)
            trainer.test(model, datamod)

            pred_batches = trainer.predict(model, datamod)
            labels_pred, labels_truth = datamod.invert_setup_on_prediction(pred_batches)

            print("labels_pred[0] =", labels_pred[0])
            print("labels_truth[0] =", labels_truth[0])

            self.assertTrue(labels_pred.shape == (100, 10))
            self.assertTrue(np.all((labels_pred >= 0) & (labels_pred <= 1)))

            sum_one_diff = np.abs(np.sum(labels_pred, axis=-1) - 1)
            print("sum_diff_mean =", np.mean(sum_one_diff))
            print("sum_diff_std =", np.std(sum_one_diff))
            self.assertTrue(np.mean(sum_one_diff) < 1e-4)

            self.assertTrue(labels_truth.shape == (100, 10))
            self.assertTrue(np.sum(labels_truth == 0.0) == len(labels_truth) * 9)
            self.assertTrue(np.sum(labels_truth == 1.0) == len(labels_truth))

    unittest.main()
