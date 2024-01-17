import logging
from pathlib import Path

import wandb
from mltools.snakemake import snakemake_main
from omegaconf import DictConfig
from wandb.apis import InternalApi as InternalWandbApi

# setup logger
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


@snakemake_main(globals().get("snakemake"))
def main(cfg: DictConfig, *run_idxs, sweep_id: str) -> None:
    wandb.login(key=cfg.wandb.api_key)
    sweep_id = Path(sweep_id).read_text()
    InternalWandbApi().stop_sweep(sweep_id, cfg.wandb.user, cfg.wandb.project)


if __name__ == "__main__":
    main()
