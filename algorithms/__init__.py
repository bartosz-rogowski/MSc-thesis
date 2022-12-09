from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class AlgorithmStrategy(ABC):
    @abstractmethod
    def find_shortest_cycle(self) -> Tuple[np.ndarray, float]:
        pass
