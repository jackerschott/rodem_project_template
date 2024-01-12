from functools import partial
import multiprocessing
from numpy.typing import NDArray
import lightning as L
import torch as T
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from typing import Any, Optional, Literal, Union, Iterable, Tuple, Dict

from .preprocs import Preprocessor
from .datasets import Dataset, MissingInfoToyDataset

class InferenceDataModule(L.LightningDataModule):
    train_set: Optional[Dataset]
    predict_set: Optional[Dataset]
    preproc: Preprocessor
    train_loader_factory: Optional[partial]
    predict_loader_factory: Optional[partial]
    split_fracs = Dict[str, float]

    x_preproc: NDArray
    y_preproc: NDArray

    datasets: Dict[str, TensorDataset]

    def __init__(self, preproc: Preprocessor, train_set: Optional[Dataset],
            predict_set: Optional[Union[Literal['use_test'], Dataset]],
            train_loader_factory: partial, predict_loader_factory: partial,
            test_frac: float, val_frac: float, split_seed: int = 0):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_set = train_set
        self.predict_set = predict_set 
        self.preproc = preproc
        self.train_loader_factory = train_loader_factory
        self.predict_loader_factory = predict_loader_factory

        self.split_fracs = dict(test=test_frac, val=val_frac)
        self.split_generator = T.Generator().manual_seed(split_seed)

    def _preproc(self, x: NDArray, y: NDArray) -> Tuple[T.Tensor, T.Tensor]:
            x_preproc = self.preproc.preprocess('hidden', x)
            x_preproc = T.tensor(x_preproc, dtype=T.float)

            y_preproc = self.preproc.preprocess('visible', y)
            y_preproc = T.tensor(y_preproc, dtype=T.float)

            return x_preproc, y_preproc

    def setup(self, stage: str):
        self.datasets = {}

        if stage == 'fit':
            x_preproc, y_preproc = self._preproc(*self.train_set.get())
            dataset = TensorDataset(x_preproc, y_preproc)
            
            split_fracs = list(self.split_fracs.values())
            split_fracs = [*split_fracs, 1 - sum(split_fracs)]
            splits = random_split(dataset, split_fracs, self.split_generator)

            i_val = list(self.split_fracs.keys()).index('val')
            self.datasets['val'] = splits[i_val]
            # from here train is only train, i.e. without val and test
            self.datasets['train'] = splits[-1]
        elif stage == 'test' or stage == 'predict' \
                and self.predict_set == 'use_test':
            _, y_preproc = self._preproc(*self.train_set.get())
            base_samples = T.randn(y_preproc.shape)
            dataset = TensorDataset(base_samples, y_preproc)

            split_fracs = list(self.split_fracs.values())
            split_fracs = [*split_fracs, 1 - sum(split_fracs)]
            splits = random_split(dataset, split_fracs, self.split_generator)

            i_test = list(self.split_fracs.keys()).index('test')
            if stage == 'test':
                self.datasets['test'] = splits[i_test]
            elif stage == 'predict':
                self.datasets['predict'] = splits[i_test]
        elif stage == 'predict':
            _, y_preproc = self._preproc(*self.predict_set.get())
            base_samples = T.randn(y_preproc.shape)
            self.datasets['predict'] = TensorDataset(base_samples, y_preproc)

    def train_dataloader(self) -> DataLoader:
        return self.train_loader_factory(self.datasets['train'], shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.train_loader_factory(self.datasets['val'], shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.train_loader_factory(self.datasets['test'], shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.predict_loader_factory(self.datasets['predict'], shuffle=False)

    def get_data_sample(self) -> DataLoader:
        if not self.train_set is None:
            x, y = self.train_set.get_init_sample()
            return T.tensor(x), T.tensor(y)
        else:
            z, y = self.predict_set.get_init_sample()
            return T.tensor(z), T.tensor(y)
            
    def invert_setup_on_prediction(self, batches: Iterable[T.Tensor]):
        pred = T.cat([batch for batch in batches]).numpy()
        return self.preproc.unpreprocess('hidden', pred)

    def save(self, path) -> None:
        state = self._get_state_dict()
        np.savez_compressed(path, **state)

    def load(self, path) -> None:
        state = np.load(path)
        self._set_state_dict(state)

if __name__ == '__main__':
    import unittest
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from .preprocs import SimplePreprocessor

    class TestDatamodule(unittest.TestCast):
        def test_setup(self):
            preproc = SimplePreprocessor()
            train_set = MissingInfoToyDataset('prior', 10_000)
            train_set.acquire()

            train_loader_factory = partial(T.utils.data.DataLoader,
                    batch_size=1024, num_workers=1, pin_memory=True)
            datamod = InferenceDataModule(preproc, train_set, None,
                    train_loader_factory, None, test_frac=0.1, val_frac=0.1)
            datamod.setup('fit')

            dataloader = datamod.train_dataloader()

            x, y = train_set.get()

            xy_preproc = T.cat([T.cat(batch, dim=-1) for batch in dataloader]).numpy()
            x_preproc, y_preproc = xy_preproc[..., 0], xy_preproc[..., 1]

            batches = [x for (x, y) in dataloader]
            x_reco = datamod.invert_setup_on_prediction(batches)

            fig, axs = plt.subplots(1, 2)

            axs[0].hist(x, density=True, histtype='step', bins=40, color='C0')
            axs[1].hist(y, density=True, histtype='step', bins=40, color='C0')

            axs[0].hist(x_preproc, density=True, histtype='step', bins=40, color='C1')
            axs[1].hist(y_preproc, density=True, histtype='step', bins=40, color='C1')

            axs[0].hist(x_reco, density=True, histtype='step', bins=40, color='C2')
            axs[1].hist(y, density=True, histtype='step', bins=40, color='C2')
            plt.show()

    unittest.main()
