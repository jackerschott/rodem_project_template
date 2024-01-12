import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from typing import Dict

import hydra
from omegaconf import DictConfig

from mltools.snakemake import snakemake_main

# setup logger
logging.basicConfig(level=logging.INFO,
        format="[%(filename)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

@snakemake_main(globals().get('snakemake'))
def main(cfg: DictConfig, prediction: str, plots: str, pred_target: str,
        **data_paths: Dict[str, str]) -> None:
    log.info('loading data and predictions')
    datasets = {}
    for name in data_paths:
        source = name.removeprefix('dataset_')
        datasets[source] = hydra.utils.instantiate(cfg.dataset,
            source=None, size=None, load_path=data_paths[name])
    prediction, = np.load(prediction).values()

    log.info('drawing plots')
    template = hydra.utils.instantiate(cfg.plot_template)
    with PdfPages(plots) as f:
        target = hydra.utils.instantiate(cfg.plot_target, fig_store=f)

        fig = target.get_figure(1)

        template.plot(fig, {pred_target: prediction}, datasets)

        target.save_figure(fig)
        plt.close(fig)

if __name__ == '__main__':
    main()
