
import weakref
import numpy as np
import pandas as pd


class TimeSeriesMatrix:

    def __init__(self):
        self.generate = self.__Generators(weakref.ref(self))
        self.normalize = self.__Normalizers(weakref.ref(self))
        self.characteristics = self.__Characteristics(weakref.ref(self))
        self._array = None

    def fromCSV(self, filename: str):
        print("Importing from CSV:", filename)

    def fromNDArray(self, array: np.ndarray):
        print("Importing from NDArray:", array)

    def toCSV(self, filename: str):
        print("Exporting to CSV:", filename)

    def toNDArray(self) -> np.ndarray:
        print("Exporting to NDArray")
        return self.array

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray):
        print("Array changed to", value)
        self._array = value

    class __Normalizers:

        def __init__(self, outer):
            self.outer = outer

        def standard(self, *args):
            print("Generating Multivariate Gaussian")
            self.outer().array = [1, 2, 3]

        def outlier(self, *args):
            print("Generating IW")
            self.outer().array = [4, 5, 6]

        def winsorization(self, *args):
            print("Generating ED")
            self.outer().array = [7, 8, 9]

    class __Generators:

        def __init__(self, outer):
            self.outer = outer

        def MVGaussian(self, *args):
            print("Generating Multivariate Gaussian")
            self.outer().array = [10, 11, 12]

        def IW(self, *args):
            print("Generating IW")
            self.outer().array = [13, 14, 15]

        def ED(self, *args):
            print("Generating ED")
            self.outer().array = [16, 17, 18]

    class __Characteristics:

        def __init__(self, outer):
            self.outer = outer

        def eigenValues(self, *args) -> np.ndarray:
            print("Fetching eigenvalues")
            eigenValues = np.ndarray((4, 4))
            return eigenValues

        def eigenVectors(self, *args) -> np.ndarray:
            print("Fetching eigenvectors")
            eigenVectors = np.ndarray((4, 2))
            return eigenVectors

        def covarianceCube(self, *args) -> np.ndarray:
            print("Fetching covariance cube")
            covarianceCube = np.ndarray((4, 4, 2))
            return covarianceCube
