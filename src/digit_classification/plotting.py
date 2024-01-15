import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import scipy.constants as cs
from sklearn.metrics import roc_curve, auc
from typing import Any, List, Dict, Tuple, Optional, Protocol

class FigureStore(Protocol):
    def savefig(self, figure: Optional[plt.Figure] = None,
            **kwargs: Any) -> None:
        ...

class PlotTarget:
    columnwidth: float
    fontsizes: Dict[str, float]
    font_code: str
    plot_label_size: float
    plot_legend_size: float
    font_family: str

    def __init__(self, fig_store: FigureStore) -> None:
        self.fig_store = fig_store
        self.figure_padding = 1.0 / 72.27 # this is one pt

    def _set_options(self):
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=self.font_code)
        plt.rc('font', size=self.plot_label_size, family=self.font_family)
        plt.rc('legend', fontsize=self.plot_label_size)
        plt.rc('axes', grid=False)
        plt.rc('axes', axisbelow=True)

    def get_plot_size(self, num_figures_per_column: int) -> Tuple[float, float]:
        plot_width = self.columnwidth \
            / (num_figures_per_column + self.plot_horizontal_gap)
        plot_height = plot_width * self.height_ratio
        return (plot_width, plot_height)

    def get_layout(self) -> mpl.layout_engine.LayoutEngine:
        return mpl.layout_engine.ConstrainedLayoutEngine(
                h_pad=0.0, w_pad=0.0, hspace=0.0, wspace=0.0)

    def get_figure(self, num_figures_per_column: int) -> plt.Figure:
        plotsize = self.get_plot_size(num_figures_per_column)
        # figsize is slightly smaller than the saved plot; we use a small
        # padding to fix objects (wrongly) cut of by constrained layout
        figsize = (plotsize[0] - 2 * self.figure_padding,
                plotsize[1] - 2 * self.figure_padding)

        layout = self.get_layout()
        return plt.figure(figsize=figsize, layout=layout)

    def save_figure(self, fig : plt.Figure) -> None:
        self.fig_store.savefig(fig, bbox_inches='tight',
                pad_inches=self.figure_padding)
        plt.close(fig)

class Monitor(PlotTarget):
    def __init__(self, fig_store: FigureStore,
            monitor_width_mm: float, fontsizes: Dict[str, float]) -> None:
        super().__init__(fig_store)

        # convert to meter then inch
        monitor_width = (0.001 * monitor_width_mm) / cs.inch

        # Half the screen width is a reasonably natural size for plots
        self.columnwidth = monitor_width / 2

        self.fontsizes = dict(
            Huge = 24.88,
            huge = 24.88,
            LARGE = 20.74,
            Large = 17.28,
            large = 14.40,
            normal = 12.00,
            small = 10.95,
            footnotesize = 10.00,
            scriptsize = 8.00,
            tiny = 6.00,
        )
        self.font_code = '\n'.join([
                r'\usepackage[bitstream-charter]{mathdesign}',
                r'\usepackage[sfdefault]{AlegreyaSans}',
                r'\usepackage{amsmath}',
        ])
        self.plot_label_size = self.fontsizes['large']
        self.plot_legend_size = self.fontsizes['large']
        self.plot_horizontal_gap = 0.0 
        self.font_family = 'sans-serif'

        self.figure_padding = 0.0
        self.height_ratio = 4/5

        self._set_options()

    def get_layout(self) -> mpl.layout_engine.LayoutEngine:
        return mpl.layout_engine.ConstrainedLayoutEngine(
                h_pad=0.1, w_pad=0.1, hspace=0.05, wspace=0.05)

class Paper(PlotTarget):
    def __init__(self, fig_store: FigureStore, columnwidth_pt: float,
            fontsizes: Dict[str, float]) -> None:
        super().__init__(fig_store)

        self.columnwidth = columnwidth_pt / 72.27

        self.fontsizes = fontsizes
        self.font_code = '\n'.join([
                r'\usepackage[bitstream-charter]{mathdesign}',
                r'\usepackage[sfdefault]{AlegreyaSans}',
                r'\usepackage{amsmath}',
        ])
        self.plot_label_size = self.fontsizes['large']
        self.plot_legend_size = self.fontsizes['large']
        self.plot_horizontal_gap = 0.0 
        self.font_family = 'sans-serif'

        self.height_ratio = 4/5

        self._set_options()

    def get_layout(self) -> mpl.layout_engine.LayoutEngine:
        return mpl.layout_engine.ConstrainedLayoutEngine(
                h_pad=0.0, w_pad=0.0, hspace=0.0, wspace=0.0)


class RocCurveTemplate:
    def __init__(self, xlabel: str = 'False Positive Rate', ylabel: str = 'True Positive Rate'):
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, fig: plt.Figure, digit: int, labels_pred: List[NDArray],
            labels_truth: List[NDArray], pred_ids: List[str]):
        # labels_pred.shape: (N, 10)
        # labels_truth.shape: (N, 10)
        assert digit in range(10)

        ax = fig.subplots(1, 1)

        for i in range(len(labels_pred)):
            prob_pred = labels_pred[i][:, digit]
            prob_truth = labels_truth[i][:, digit]

            fpr, tpr, _ = roc_curve(prob_truth, prob_pred)
            area = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{pred_ids[i]} (AUC = {area:.2f})')
        
        ax.plot([0, 1], [0, 1], color='lightgray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(f'Classification of Digit {digit}')
        ax.legend(loc="lower right", frameon=False)

if __name__ == '__main__':
    import unittest

    class TestRocCurveTemplate(unittest.TestCase):
        def test(self):
            pass
            
    unittest.main()
