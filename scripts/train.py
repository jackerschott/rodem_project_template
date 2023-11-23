"""Basic training script."""

import logging

import hydra
import lightning.pytorch as pl
import pyrootutils
import torch as T
from omegaconf import DictConfig

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.hydra_utils import (
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    """Load train and save a model."""

    log.info("Setting up full job config")
    if cfg.full_resume:
        cfg = reload_original_config()
    print_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.precision}")
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating the model")
    model = hydra.utils.instantiate(cfg.model, data_sample=datamodule.get_data_sample())

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = hydra.utils.instantiate(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = hydra.utils.instantiate(cfg.loggers)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)
        log.info(model)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Starting training!")
    trainer.fit(model, datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
