import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mltools.snakemake import snakemake_main
from model_indep_unfolding.datamodules.datasets import Dataset

# setup logger
logging.basicConfig(level=logging.INFO,
        format="[%(filename)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

@snakemake_main(globals().get('snakemake'))
def main(cfg: DictConfig, dataset: str) -> None:
    print(cfg.dataset)
    dataset_to_acquire: Dataset = hydra.utils.instantiate(cfg.dataset)

    log.info('Acquiring data')
    dataset_to_acquire.acquire()

    log.info('Saving data')
    dataset_to_acquire.save(dataset)

if __name__ == '__main__':
    main()
