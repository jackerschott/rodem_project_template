from abc import ABC, abstractmethod
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy.typing import NDArray
from scipy.stats import truncnorm as scipy_truncnorm
from typing import Any, Tuple, Dict, Optional

class Dataset(ABC):
    source: str
    size: int
    x: NDArray
    y: NDArray

    def __init__(self, source: Optional[str] = None, size: Optional[int] = None,
            load_path: Optional[str] = None) -> None:
        if source is None and size is None and not load_path is None:
            self.load(load_path)
            self.size = len(self.x)
        elif not source is None and not size is None and load_path is None:
            self.source = source
            self.size = size
        else:
            assert False

    @abstractmethod
    def acquire(self) -> None:
        ...

    @abstractmethod
    def get(self) -> Tuple[NDArray, NDArray]:
        ...

    @abstractmethod
    def get_init_sample(self) -> Tuple[NDArray, NDArray]:
        ...

    def save(self, filename: str) -> None:
        np.savez_compressed(filename, self.source, self.x, self.y)

    def load(self, filename: str) -> None:
        self.source, self.x, self.y = np.load(filename).values()
        self.source = str(self.source)

#class WellToyDataset(Dataset):
#    DATA_DIM = 2
#
#    def __init__(self, source: str, size: Optional[int], load_path: Optional[str]) -> None:
#        super().__init__(source, size, load_path)
#
#    def acquire(self) -> None:
#        if self.source == 'theory':
#            self.x = np.stack([truncnormal(0.5, 0.5, 0.0, 1.0, self.size),
#                truncnormal(0.5, 0.1, 0.0, 1.0, self.size)], axis=-1)
#        elif self.source == 'data':
#            self.x = np.stack([truncnormal(0.2, 0.4, 0.0, 1.0, self.size),
#                truncnormal(0.3, 0.2, 0.0, 1.0, self.size)], axis=-1)
#        elif self.source == 'prior':
#            self.x = np.stack([np.random.uniform(0.0, 1.0, size=self.size),
#                np.random.uniform(0.0, 1.0, size=self.size)], axis=-1)
#        else:
#            assert False
#
#        self.y = self._hidden2visible(self.x)
#
#    def _hidden2visible(self, x: NDArray) -> NDArray:
#        swap = np.array(np.random.randint(0, 2, size=len(x)), dtype=bool)
#
#        y = np.full_like(x, np.nan)
#        y[swap] = x[swap][:, [1, 0]]
#        y[~swap] = x[~swap][:, [0, 1]]
#        return y
#
#    def get(self) -> Tuple[NDArray, NDArray]:
#        return self.x, self.y
#
#    def visualize(self, filename: str) -> None:
#        raise NotImplementedError

class MissingInfoToyDataset(Dataset):
    DATA_DIM = 1

    def __init__(self, source: str = None, size: Optional[int] = None,
            load_path: Optional[str] = None) -> None:
        super().__init__(source, size, load_path)

    def _generate(self, size) -> None:
        if self.source == 'theory':
            x = truncnormal(-0.03, 0.2, -1.0, 1.0,
                    size=size).reshape(-1, 1)
        elif self.source == 'data':
            x = truncnormal(0.1, 0.1, -1.0, 1.0,
                    size=size).reshape(-1, 1)
        elif self.source == 'prior':
            x = np.random.uniform(-1.0, 1.0,
                    size=size).reshape(-1, 1)
        else:
            assert False

        y = self._hidden2visible(x)
        return x, y
        

    def acquire(self) -> None:
        self.x, self.y = self._generate(self.size)

    def _hidden2visible(self, x: NDArray) -> NDArray:
        y = np.full_like(x, np.nan)
        y[x >= 0] = 3.0 * x[x >= 0] + 2.0
        y[x < 0] = 2.0
        y += np.random.normal(0.0, 0.3, size=x.shape)
        return y

    def get(self) -> Tuple[NDArray, NDArray]:
        return self.x, self.y

    def get_init_sample(self) -> Tuple[NDArray, NDArray]:
        return self._generate(1)


def truncnormal(loc: float, scale: float, a: float, b: float, size: int) -> NDArray:
    a, b = (a - loc) / scale, (b - loc) / scale
    return scipy_truncnorm.rvs(a, b, size=size) * scale + loc
