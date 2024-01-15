from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any

class Preprocessor(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, data_type: str, x: NDArray) -> NDArray:
        ...

    @abstractmethod
    def unpreprocess(self, data_type: str, x: NDArray) -> NDArray:
        ...


class SimplePreprocessor(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    def _preproc_labels(self, labels: NDArray) -> NDArray:
        return labels

    def _preproc_data_imgs(self, digit_img: NDArray) -> NDArray:
        assert digit_img.shape[-1] == digit_img.shape[-2]
        img_size = digit_img.shape[-1]
        return digit_img.reshape(len(digit_img), 1, img_size, img_size)

    def preprocess(self, data_type: str, x: NDArray) -> NDArray:
        if data_type == 'labels':
            return self._preproc_labels(x)
        elif data_type == 'digit_imgs':
            return self._preproc_data_imgs(x)
        else:
            assert False

    def unpreprocess(self, data_type: str, x: NDArray) -> NDArray:
        assert data_type == 'labels'
        return x

if __name__ == '__main__':
    import unittest

    class TestSimplePreprocessor(unittest.TestCase):
        def test_consistency(self):
            preproc = SimplePreprocessor()

            labels = np.random.rand(10, 10)

            preproc = SimplePreprocessor()
            labels_preproc = preproc.preprocess('labels', labels)
            labels_reco = preproc.unpreprocess('labels', labels_preproc)

            reco_err = np.abs(labels_preproc - labels_reco)
            self.assertTrue(np.mean(reco_err) < 1e-4)

            print('reco_err_mean:', np.mean(reco_err))
            print('reco_err_std:', np.std(reco_err))

    unittest.main()
