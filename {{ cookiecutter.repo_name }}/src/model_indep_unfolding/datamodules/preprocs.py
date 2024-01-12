from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any

class Preprocessor(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, level: str, x: NDArray) -> NDArray:
        ...

    @abstractmethod
    def unpreprocess(self, level: str, x: NDArray) -> NDArray:
        ...


class SimplePreprocessor(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, level: str, x: NDArray) -> NDArray:
        if level == 'hidden':
            assert not np.any((x < -1.0) | (x > 1.0))
            return np.arctanh(x)
        elif level == 'visible':
            return x
        else:
            assert False

    def unpreprocess(self, level: str, x: NDArray) -> NDArray:
        if level == 'hidden':
            return np.tanh(x)
        elif level == 'visible':
            return x
        else:
            assert False
