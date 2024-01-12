from omegaconf import DictConfig

config: DictConfig

def set_config(cfg: DictConfig) -> None:
    global config
    config = cfg

def get_config() -> DictConfig:
    global config
    return config
