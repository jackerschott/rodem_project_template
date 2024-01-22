import logging
import os
import re
from pathlib import Path

import hydra
import lightning as L
import torch as T
import wandb
from omegaconf import DictConfig

# setup logger
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


def find_max_epoch_checkpoint(checkpoints_pattern: str, checkpoints_path: str):
    if not os.path.exists(checkpoints_path):
        return None

    checkpoints_pattern = checkpoints_pattern.format(epoch="([0-9]+)")
    epoch_checkpoint_paths = [
        path
        for path in os.listdir(checkpoints_path)
        if re.search(checkpoints_pattern, path)
    ]
    if len(epoch_checkpoint_paths) == 0:
        return None

    max_filename = max(
        epoch_checkpoint_paths,
        key=lambda path: re.search(checkpoints_pattern, path).group(1),
    )
    return Path(checkpoints_path) / max_filename


def train(
    log,
    cfg: DictConfig,
) -> (L.Trainer, L.LightningModule, L.LightningDataModule):
    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.precision}")
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the datamodule")
    dataset = hydra.utils.instantiate(cfg.dataset)
    datamodule = hydra.utils.instantiate(
        cfg.datamodule, train_set=dataset, predict_set=None
    )

    log.info("Instantiating the model")
    model = hydra.utils.instantiate(cfg.model,
            data_sample=datamodule.get_data_sample())

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating the trainer")
    # we want to have (easy) access to callbacks and logger,
    # so we need to convert to a list for the trainer constructor
    checkpoints_pattern = cfg.trainer.callbacks.model_checkpoint.filename
    cfg.trainer.callbacks = list(cfg.trainer.callbacks.values())
    cfg.trainer.logger = list(cfg.trainer.logger.values())
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Starting training!")
    ckpt_path = (
        find_max_epoch_checkpoint(checkpoints_pattern, checkpoints_path)
        if cfg.auto_resume
        else None
    )
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    checkpoint_callback = [
        cb
        for cb in trainer.callbacks
        if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint)
    ][0]

    if cfg.model_save_path:
        best_model_path_rel = os.path.relpath(
            checkpoint_callback.best_model_path,
            Path(cfg.model_save_path).parent
        )
        os.symlink(best_model_path_rel, cfg.model_save_path)

    return trainer, model, datamodule


@hydra.main(config_path='../config', config_name='train_model', version_base=None)
def main(cfg: DictConfig) -> None:
    wandb.login(key=cfg.wandb.api_key)

    wandb_run_id = (
        Path(cfg.wandb.run_id_path).read_text()
        if os.path.exists(cfg.wandb.run_id_path)
        else None
    )
    wandb.init(
        job_type="train",
        dir=cfg.wandb.save_dir,
        project=cfg.wandb.project,
        entity=cfg.wandb.user,
        tags=cfg.wandb.tags,
        name=cfg.wandb.run_name,
        resume=("must" if wandb_run_id else None),
        id=wandb_run_id,
    )
    Path(cfg.wandb.run_id_path).write_text(wandb.run.id)

    train(log, cfg)


if __name__ == "__main__":
    main()
