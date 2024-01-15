from functools import partial
import multiprocessing
import numpy as np
from numpy.typing import NDArray
import lightning as L
import torch as T
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from typing import Any, Optional, Literal, Union, Iterable, Tuple, Dict

from .preprocs import Preprocessor
from .datasets import Dataset

import matplotlib.pyplot as plt

class LabelledDigitsModule(L.LightningDataModule):
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

    def _preproc(self, labels: NDArray, digit_imgs: NDArray) \
            -> Tuple[T.Tensor, T.Tensor]:
        # one-hot encoding can only be undone for 1 or 0 labels
        # not for the network output; hence it doesn't belong in the preprocessor
        labels_encoded = np.eye(10)[labels]

        labels_preproc = self.preproc.preprocess('labels', labels_encoded)
        labels_preproc = T.tensor(labels_preproc, dtype=T.float)

        digit_imgs_preproc = self.preproc.preprocess('digit_imgs', digit_imgs)
        digit_imgs_preproc = T.tensor(digit_imgs_preproc, dtype=T.float)

        return labels_preproc, digit_imgs_preproc

    def setup(self, stage: str):
        self.datasets = {}

        if stage == 'fit':
            labels_preproc, digit_imgs_preproc = self._preproc(*self.train_set.get())
            dataset = TensorDataset(labels_preproc, digit_imgs_preproc)
            
            split_fracs = list(self.split_fracs.values())
            split_fracs = [*split_fracs, 1 - sum(split_fracs)]
            splits = random_split(dataset, split_fracs, self.split_generator)

            i_val = list(self.split_fracs.keys()).index('val')
            self.datasets['val'] = splits[i_val]
            # from here train is only train, i.e. without val and test
            self.datasets['train'] = splits[-1]
        elif stage == 'test' or stage == 'predict' \
                and self.predict_set == 'use_test':
            labels_preproc, digit_imgs_preproc = self._preproc(*self.train_set.get())
            dataset = TensorDataset(labels_preproc, digit_imgs_preproc)

            split_fracs = list(self.split_fracs.values())
            split_fracs = [*split_fracs, 1 - sum(split_fracs)]
            splits = random_split(dataset, split_fracs, self.split_generator)

            i_test = list(self.split_fracs.keys()).index('test')
            if stage == 'test':
                self.datasets['test'] = splits[i_test]
            elif stage == 'predict':
                self.datasets['predict'] = splits[i_test]
        elif stage == 'predict':
            labels_preproc, digit_imgs_preproc = self._preproc(*self.predict_set.get())
            self.datasets['predict'] = TensorDataset(labels_preproc, digit_imgs_preproc)

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
            label, digit_img = self.train_set.get_init_sample()
        else:
            label, digit_img = self.predict_set.get_init_sample()

        label_preproc, digit_img_preproc = self._preproc(label, digit_img)
        return label_preproc, digit_img_preproc
            
    def invert_setup_on_prediction(self, batches: Iterable[T.Tensor]):
        labels_pred = T.cat([x for x, _ in batches]).numpy()
        labels_truth = T.cat([x for _, x in batches]).numpy()

        labels_pred = self.preproc.unpreprocess('labels', labels_pred)
        labels_truth = self.preproc.unpreprocess('labels', labels_truth)
        return labels_pred, labels_truth

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
    from .datasets import MNISTDataset

    class TestLabelledDigitsModule(unittest.TestCase):
        def test_setup(self):
            preproc = SimplePreprocessor()
            train_set = MNISTDataset(10_000)
            train_set.acquire(tmp_save_dir='mnist')

            train_loader_factory = partial(T.utils.data.DataLoader,
                    batch_size=1024, num_workers=1, pin_memory=True)
            datamod = LabelledDigitsModule(preproc, train_set, None,
                    train_loader_factory, None, test_frac=0.1, val_frac=0.1)
            datamod.setup('fit')

            dataloader = datamod.train_dataloader()

            for _labels, _digit_imgs in dataloader:
                label, digit_img = _labels[0], _digit_imgs[0]

            self.assertTrue(label.shape == (10,))
            self.assertTrue(T.sum(label == 0.0) == 9 and T.sum(label == 1.0) == 1)

            self.assertTrue(digit_img.shape == (1, 28, 28))
            self.assertTrue(T.all((digit_img >= 0.0) & (digit_img <= 1.0)))
            self.assertTrue(T.all((digit_img >= 0.0) & (digit_img <= 1.0)))

        def test_get_data_sample(self):
            preproc = SimplePreprocessor()
            train_set = MNISTDataset(10_000)
            train_set.acquire(tmp_save_dir='mnist')

            train_loader_factory = partial(T.utils.data.DataLoader,
                    batch_size=1024, num_workers=1, pin_memory=True)
            datamod = LabelledDigitsModule(preproc, train_set, None,
                    train_loader_factory, None, test_frac=0.1, val_frac=0.1)

            label, digit_img = datamod.get_data_sample()
            self.assertTrue(label.shape == (1, 10))
            self.assertTrue(T.sum(label == 0.0) == 9 and T.sum(label == 1.0) == 1)

            self.assertTrue(digit_img.shape == (1, 1, 28, 28))
            self.assertTrue(T.all((digit_img >= 0.0) & (digit_img <= 1.0)))
            self.assertTrue(T.all((digit_img >= 0.0) & (digit_img <= 1.0)))

    unittest.main()
