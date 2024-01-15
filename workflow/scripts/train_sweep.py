from functools import partial
import logging
from pathlib import Path
from typing import List

from omegaconf import OmegaConf, DictConfig
import wandb

from mltools.snakemake import snakemake_main
from train import train

# setup logger
logging.basicConfig(level=logging.INFO,
        format='[%(filename)s] %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def train_test_under_agent(cfg: DictConfig,
        dataset: str, checkpoints_base_path: str):
    wandb.init(job_type='train_sweep', project=cfg.wandb.project,
            entity=cfg.wandb.user, tags=cfg.wandb.tags, name=cfg.wandb.name)

    for param_name, config_key in cfg.sweep.param_config_keys.items():
        OmegaConf.update(cfg, config_key, wandb.config[param_name])
        
    # Wandb Runs that are part of a sweeps cannot be resumed
    # (see https://docs.wandb.ai/guides/runs/resuming)
    trainer, model, datamodule = train(log, cfg, dataset, model_path=None,
            checkpoints_path=(Path(checkpoints_base_path) / wandb.run.id),
            auto_resume=False)

    log.info('Starting testing!')
    trainer.test(model, datamodule)

@snakemake_main(globals().get('snakemake'))
def main(cfg: DictConfig, dataset: str, sweep_id: str,
        job_completion_marker: str, checkpoints_base_path: str,
        trainer_default_root_dir: str) -> None:
    cfg.trainer.default_root_dir = trainer_default_root_dir

    wandb.login(key=cfg.wandb.api_key)

    sweep_id = Path(sweep_id).read_text()
    train_under_test_agent_ = partial(train_test_under_agent,
            cfg, dataset, checkpoints_base_path)
    wandb.agent(sweep_id, train_under_test_agent_, cfg.wandb.user,
            cfg.wandb.project, count=cfg.sweep.runs_per_job_count)

    Path(job_completion_marker).touch()

if __name__ == '__main__':
    main()
