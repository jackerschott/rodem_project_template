import copy
from functools import partial
from typing import Callable, Mapping, Tuple

import lightning.pytorch as pl
import torch as T
import wandb
from torchvision.utils import make_grid

from mltools.mltools.cnns import UNet
from mltools.mltools.diffusion import append_dims, c_values
from mltools.mltools.modules import CosineEncodingLayer, IterativeNormLayer
from mltools.mltools.torch_utils import ema_param_sync, get_sched


class UNetDiffusion(pl.LightningModule):
    """A generative model which uses the diffusion process on an image input."""

    def __init__(
        self,
        *,
        data_sample: tuple,
        cosine_config: Mapping,
        normaliser_config: Mapping,
        unet_config: Mapping,
        optimizer: partial,
        sched_config: Mapping,
        min_sigma: float = 0,
        max_sigma: float = 80.0,
        ema_sync: float = 0.999,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sampler_function: Callable | None = None,
        sigma_function: Callable | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Break down the data sample to get the important dimensions
        self.inpt_dim = data_sample[0].shape

        # Class attributes
        self.ema_sync = ema_sync
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.p_mean = p_mean
        self.p_std = p_std

        # The cosine encoder for the diffusion sigma parameter
        self.sigma_encoder = CosineEncodingLayer(
            inpt_dim=1, min_value=0, max_value=max_sigma, **cosine_config
        )

        # The layer which normalises the input data
        self.normaliser = IterativeNormLayer(
            self.inpt_dim, dims=(1, 2), **normaliser_config
        )

        # The base UNet
        self.net = UNet(
            inpt_size=self.inpt_dim[1:],
            inpt_channels=self.inpt_dim[0],
            outp_channels=self.inpt_dim[0],
            ctxt_dim=self.sigma_encoder.outp_dim,
            **unet_config,
        )

        # A copy of the network which will sync with an exponential moving average
        self.ema_net = copy.deepcopy(self.net)
        self.ema_net.requires_grad_(False)

        # Sampler to run in the validation/testing loop
        self.sampler_function = sampler_function
        self.sigma_function = sigma_function

        # Initial noise for running the visualisation during validation
        self.register_buffer(
            "initial_noise", T.randn((5, *self.inpt_dim)) * self.max_sigma
        )

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")

    def forward(
        self,
        noisy_data: T.Tensor,
        sigmas: T.Tensor,
    ) -> T.Tensor:
        """Return the denoised data from a given sigma value."""

        # Get the c values for the data scaling
        c_in, c_out, c_skip = c_values(append_dims(sigmas, noisy_data.dim()))

        # Scale the inputs and pass through the network
        outputs = self.get_outputs(c_in * noisy_data, sigmas)

        # Get the denoised output by passing the scaled input through the network
        return c_skip * noisy_data + c_out * outputs

    def get_outputs(
        self,
        noisy_data: T.Tensor,
        sigmas: T.Tensor,
    ) -> T.Tensor:
        """Pass through the model, corresponds to F_theta in the Karras paper."""

        # Use the appropriate network for training or validation
        if self.training or not self.ema_sync:
            network = self.net
        else:
            network = self.ema_net

        # Encode the sigmas and combine with existing context info
        ctxt = self.sigma_encoder(sigmas)

        # Use the selected network to esitmate the noise present in the data
        return network(noisy_data, ctxt=ctxt)

    def _shared_step(self, sample: tuple) -> Tuple[T.Tensor, T.Tensor]:
        """Shared step used in both training and validaiton."""

        # Unpack the sample tuple
        data = sample[0]  # pytorch datasets also return label etc which we dont want

        # Pass through the normalisers
        data = self.normaliser(data)

        # Sample sigmas using the Karras method of a log normal distribution
        sigmas = T.randn(size=(data.shape[0], 1), device=self.device)
        sigmas.mul_(self.p_std).add_(self.p_mean).exp_()
        sigmas.clamp_(self.min_sigma, self.max_sigma)

        # Get the c values for the data scaling
        sigmas_with_dim = append_dims(sigmas, data.dim())
        c_in, c_out, c_skip = c_values(sigmas_with_dim)

        # Sample from N(0, sigma**2)
        noises = T.randn_like(data) * sigmas_with_dim

        # Make the noisy samples by mixing with the real data
        noisy_data = data + noises

        # Pass through the just the base network (manually scale with c values)
        output = self.get_outputs(c_in * noisy_data, sigmas)

        # Get the karras effective target
        scaled_target = (data - c_skip * noisy_data) / c_out

        # Return the denoising loss
        return (output - scaled_target).square().mean()

    def training_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        loss = self._shared_step(sample)
        self.log("train/total_loss", loss)
        ema_param_sync(self.net, self.ema_net, self.ema_sync)
        return loss

    def validation_step(self, sample: tuple, batch_idx: int) -> None:
        loss = self._shared_step(sample)
        self.log("valid/total_loss", loss)

        # Once per epoch and if wandb is running, run full generation and uploa
        if batch_idx == 0 and wandb.run is not None:
            gen_images = self.full_generation(initial_noise=self.initial_noise)
            gen_images = wandb.Image(make_grid(gen_images.clamp(0, 1)))
            wandb.log({"gen_images": gen_images}, commit=False)

    @T.no_grad()
    def full_generation(self, initial_noise: T.Tensor) -> T.Tensor:
        """Fully generate a batch of data from noise."""

        # Generate the sigma values (Descending!!)
        sigmas = self.sigma_function(self.min_sigma, self.max_sigma)

        # Run the sampler
        outputs = self.sampler_function(
            model=self,
            x=initial_noise,
            sigmas=sigmas,
        )

        # Return the output unormalised again
        return self.normaliser.reverse(outputs)

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate scheduler for this model."""

        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.net.parameters())

        # Use mltools to initialise the scheduler (allows cyclic-epoch sync)
        sched = get_sched(
            self.hparams.sched_config.mltools,
            opt,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            max_epochs=self.trainer.max_epochs,
        )

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.sched_config.lightning},
        }
