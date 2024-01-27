import logging

import hydra
from omegaconf import DictConfig

# setup logger
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

@hydra.main(config_path="../config",
        config_name="acquire_data", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_to_acquire = hydra.utils.instantiate(cfg.dataset)

    log.info("Acquiring data")
    dataset_to_acquire.acquire()

    log.info("Saving data")
    dataset_to_acquire.save(cfg.save_path)

if __name__ == "__main__":
    main()
