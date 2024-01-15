import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from typing import List, Dict

import hydra
from omegaconf import DictConfig

from mltools.snakemake import snakemake_main

# setup logger
logging.basicConfig(level=logging.INFO,
        format="[%(filename)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

@snakemake_main(globals().get('snakemake'))
def main(cfg: DictConfig, *predictions: List[str],
        roc_curves: str, pred_ids: List[str]) -> None:
    log.info('loading data and predictions')

    labels_pred, labels_truth = [], []
    for pred_path in predictions:
        (pred, truth), = np.load(pred_path).values()
        labels_pred.append(pred)
        labels_truth.append(truth)

    log.info('drawing plots')
    template = hydra.utils.instantiate(cfg.plot_template)
    with PdfPages(roc_curves) as f:
        target = hydra.utils.instantiate(cfg.plot_target, fig_store=f)

        for digit in range(10):
            fig = target.get_figure(1)
            template.plot(fig, digit, labels_pred, labels_truth, pred_ids)
            target.save_figure(fig)

        plt.close(fig)

if __name__ == '__main__':
    main()
