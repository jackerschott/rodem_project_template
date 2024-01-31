import logging
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig

from mltools.utils import load_nested_array_dict_from_h5

# setup logger
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="plot", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("loading data and predictions")
    dataset = hydra.utils.instantiate(cfg.dataset)
    for pred_path in cfg.pred_paths:
        pred = load_nested_array_dict_from_h5(cfg.pred_paths)
    
    #labels_pred, labels_truth = [], []
    #for pred_path in predictions:
    #    ((pred, truth),) = np.load(pred_path).values()
    #    labels_pred.append(pred)
    #    labels_truth.append(truth)

    #log.info("drawing plots")
    #template = hydra.utils.instantiate(cfg.plot_template)
    #with PdfPages(roc_curves) as f:
    #    target = hydra.utils.instantiate(cfg.plot_target, fig_store=f)

    #    for digit in range(10):
    #        # use 3 figures per column to arrange
    #        # ROC curves for 9 digits in a 3x3 grid
    #        fig = target.get_figure(3)
    #        template.plot(fig, digit, labels_pred, labels_truth, pred_ids)
    #        target.save_figure(fig)

    #    plt.close(fig)


if __name__ == "__main__":
    main()
