
import weakref
import numpy as np
import pandas as pd

from .MatrixGenerators import MatrixGeneratorAdapter
from .MatrixNormalizers import MatrixNormalizerAdapter
from .MatrixSamplers import MatrixSamplerAdapter


class TimeSeriesMatrix:

    def __init__(self):
        self.generate = MatrixGeneratorAdapter(weakref.ref(self))
        self.normalize = MatrixNormalizerAdapter(weakref.ref(self))
        self.characteristics = MatrixSamplerAdapter(weakref.ref(self))
        self._array = None

    def from_CSV(self, filepath: str):
        print("Importing from CSV:", filepath)
        self.array = pd.read_csv(filepath, header=None).values.T

    def from_ndarray(self, array: np.ndarray):
        print("Importing from NDArray")
        self.array = array

    def to_CSV(self, filepath: str):
        print("Exporting to CSV:", filepath)
        pd.DataFrame(self._array.T).to_csv(filepath, header=None, index=None)

    def to_ndarray(self) -> np.ndarray:
        print("Exporting to NDArray")
        return self.array

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray):
        self._array = value


