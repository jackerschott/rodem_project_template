import logging
import numpy as np
import tempfile

import hydra
import lightning as L
from omegaconf import DictConfig
from snakemake.script import Snakemake as SnakemakeContext
import torch as T

from mltools.snakemake import snakemake_main

# setup logger
logging.basicConfig(level=logging.INFO,
        format="[%(filename)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

@snakemake_main(globals().get('snakemake'))
def main(cfg: DictConfig, dataset: str, model: str, prediction: str) -> None:
    if cfg.seed:
        log.info(f'Setting seed to: {cfg.seed}')
        L.seed_everything(cfg.seed, workers=True)

    log.info(f'Setting matrix precision to: {cfg.precision}')
    T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the datamodule")
    dataset = hydra.utils.instantiate(cfg.dataset, size=None, load_path=dataset)
    datamodule = hydra.utils.instantiate(cfg.datamodule,
            train_set=dataset, predict_set='use_test')

    log.info("Loading model from checkpoint")
    ModelClass = hydra.utils.get_class(cfg.model._target_)
    model = ModelClass.load_from_checkpoint(model,
            data_sample=datamodule.get_data_sample(),
            optimizer_factory=cfg.model.optimizer_factory)

    log.info('Instantiating the trainer')
    # we are just predicting, so we don't care about logs
    with tempfile.TemporaryDirectory() as tempdir:
        cfg.trainer.default_root_dir = tempdir
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer,
            logger=None, callbacks=None)

    log.info("Predicting")
    batches = trainer.predict(model, datamodule)
    pred = datamodule.invert_setup_on_prediction(batches)

    log.info("Save prediction")
    np.savez_compressed(prediction, pred)

if __name__ == '__main__':
    main()
