import logging
import pickle
import tempfile

import hydra
import lightning as L
import torch as T
from omegaconf import DictConfig

from mltools.mltools.utils import save_nested_array_dict_to_h5

# setup logger
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Setting matrix precision to: {cfg.trainenv.precision}")
    T.set_float32_matmul_precision(cfg.trainenv.precision)

    log.info("Instantiating the datamodule")
    dataset = hydra.utils.instantiate(cfg.dataset)
    datamodule = hydra.utils.instantiate(
        cfg.datamodule, train_set=None, predict_set=dataset
    )

    log.info("Loading model from checkpoint")
    ModelClass = hydra.utils.get_class(cfg.model._target_)
    model = ModelClass.load_from_checkpoint(cfg.model_checkpoint_path)

    log.info("Instantiating the trainer")
    # we are just predicting, so we don't care about logs
    with tempfile.TemporaryDirectory() as tempdir:
        trainer = L.Trainer(
            enable_progress_bar=True, logger=None, default_root_dir=tempdir
        )

        log.info("Predicting")
        batches = trainer.predict(model, datamodule)

    pred = datamodule.invert_setup_on_prediction(batches)

    log.info("Save prediction")
    save_nested_array_dict_to_h5(cfg.predictions_save_path, pred)

if __name__ == "__main__":
    main()
