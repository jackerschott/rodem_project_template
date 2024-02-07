from functools import partial
from typing import Tuple

import lightning as L
import torch as T
import torch.nn as nn


class ConvClassifier(L.LightningModule):
    IMG_SIZE = 28
    DIGIT_COUNT = 10

    network: nn.Module
    optimizer_factory: partial

    def __init__(
        self,
        mock_sample: Tuple[T.Tensor, T.Tensor],
        conv_blocks_1: int,
        conv_blocks_2: int,
        hidden_conv_channels: int,
        mlp_depth: int,
        optimizer_factory: partial,
    ):
        super().__init__()
        self.save_hyperparameters()

        label_sample, digit_img_sample = mock_sample

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
            nn.Linear(mlp_width, self.DIGIT_COUNT),
            nn.Softmax(dim=-1),
        )

        self.optimizer_factory = optimizer_factory

    def configure_optimizers(self) -> T.optim.Optimizer:
        return self.optimizer_factory(self.network.parameters())

    def forward(self, digit_imgs: T.Tensor) -> T.Tensor:
        # digit_imgs must regard their values as one channel for convolution blocks;
        # also, we want to normalize them to [0, 1]
        digit_imgs = digit_imgs.unsqueeze(1) / 256
        return self.network(digit_imgs)

    def _development_step(
        self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int
    ) -> T.Tensor:
        labels, digit_imgs = batch
        labels_encoded = T.eye(self.DIGIT_COUNT, device=labels.device)[labels]

        labels_pred = self.forward(digit_imgs)
        return nn.functional.binary_cross_entropy(labels_pred, labels_encoded)

    def training_step(self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int) -> None:
        loss = self._development_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)

    def validation_step(self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int) -> None:
        loss = self._development_step(batch, batch_idx)
        self.log(
            "valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def predict_step(
        self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int
    ) -> T.Tensor:
        _, digit_imgs = batch
        return self.forward(digit_imgs)


if __name__ == "__main__":
    import tempfile
    import unittest

    from omegaconf import OmegaConf

    from ...data.mnist import MNISTDataModule

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
            with tempfile.TemporaryDirectory() as load_path:
                dev_set_conf = OmegaConf.create(
                    dict(
                        _target_="digit_classification.data.mnist.MNISTDataset",
                        load_path=load_path,
                        train=True,
                        size=2,
                    )
                )
                predict_set_conf = OmegaConf.create(
                    dict(
                        _target_="digit_classification.data.mnist.MNISTDataset",
                        load_path=load_path,
                        train=False,
                        size=2,
                    )
                )

                dev_loader_conf = OmegaConf.create(dict(batch_size=1))
                predict_loader_conf = OmegaConf.create(dict(batch_size=1))

                datamod = MNISTDataModule(
                    dev_set_conf=dev_set_conf,
                    predict_set_conf=predict_set_conf,
                    val_frac=0.5,
                    dev_loader_conf=dev_loader_conf,
                    predict_loader_conf=predict_loader_conf,
                )

            model = ConvClassifier(
                datamod.mock_sample(),
                conv_blocks_1=2,
                conv_blocks_2=2,
                hidden_conv_channels=8,
                mlp_depth=3,
                optimizer_factory=partial(T.optim.Adam, lr=1.0e-3),
            )

            # TODO: use unit testing capabilities of lightning
            trainer = L.Trainer(
                max_epochs=1,
                enable_checkpointing=False,
                enable_progress_bar=True,
                enable_model_summary=True,
            )
            trainer.fit(model, datamod)

            pred_batches = trainer.predict(model, datamod)
            labels_pred = datamod.invert_setup_on_prediction(pred_batches)

            print("labels_pred.shape =", labels_pred.shape)

    unittest.main()
