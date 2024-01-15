import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mltools.snakemake import snakemake_main

# setup logger
logging.basicConfig(level=logging.INFO,
        format="[%(filename)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

@snakemake_main(globals().get('snakemake'))
def main(cfg: DictConfig, dataset: str) -> None:
    dataset_to_acquire = hydra.utils.instantiate(cfg.dataset)

    log.info('Acquiring data')
    dataset_to_acquire.acquire()

    log.info('Saving data')
    dataset_to_acquire.save(dataset)

if __name__ == '__main__':
    main()
