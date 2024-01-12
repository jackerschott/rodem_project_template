import logging
import os
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from mltools.snakemake import snakemake_main

# setup logger
logging.basicConfig(level=logging.INFO,
        format='[%(filename)s] %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

@snakemake_main(globals().get('snakemake'))
def main(cfg: DictConfig, sweep_id: str, sweep_name: str) -> None:
    wandb.login(key=cfg.wandb.api_key)
    sweep_config = OmegaConf.to_container(cfg.sweep.config)
    sweep_config['name'] = sweep_name
    sweep_id_ = wandb.sweep(sweep_config, cfg.wandb.user, cfg.wandb.project)

    Path(sweep_id).write_text(sweep_id_)
    
if __name__ == '__main__':
    main()
