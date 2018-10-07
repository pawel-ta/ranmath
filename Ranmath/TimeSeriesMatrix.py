
import weakref
import numpy as np
import pandas as pd
import scipy.linalg as la


class TimeSeriesMatrix:

    def __init__(self):
        self.generate = self.__Generators(weakref.ref(self))
        self.normalize = self.__Normalizers(weakref.ref(self))
        self.characteristics = self.__Characteristics(weakref.ref(self))
        self._array = None

    def fromCSV(self, filepath: str):
        print("Importing from CSV:", filepath)
        self.array = pd.read_csv(filepath, header=None).values
        #self.array = pd.DataFrame.from_csv(filepath).as_matrix()

    def fromNDArray(self, array: np.ndarray):
        print("Importing from NDArray")
        self.array = array

    def toCSV(self, filepath: str):
        print("Exporting to CSV:", filepath)
        pd.DataFrame(self._array).to_csv(filepath, header=None, index=None)

    def toNDArray(self) -> np.ndarray:
        print("Exporting to NDArray")
        return self.array

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray):
        self._array = value

    class __Normalizers:

        def __init__(self, outer):
            self.outer = outer

        def standard(self, *args):
            print("Standard normalization")
            self.outer().array = [1, 2, 3]

        def outlier(self, *args):
            print("Outlier normalization")
            self.outer().array = [4, 5, 6]

        def winsorization(self, *args):
            print("Winsorization")
            self.outer().array = [7, 8, 9]

    class __Generators:

        def __init__(self, outer):
            self.outer = outer

        def mvGaussian(self, C: np.ndarray, A: np.ndarray):
            print("Generating using Multivariate Gaussian")

            N, T = C.shape, A.shape
            if N[0] != N[1] or T[0] != T[1]:
                raise ValueError('C and A should be square matrices')
            N, T = N[0], T[0]

            C_root = la.sqrtm(C).real
            A_root = la.sqrtm(A).real
            random_matrix = np.random.normal(size=(N, T))

            self.outer().array = C_root @ random_matrix @ A_root

        def inverseWishart(self, number_of_assets, number_of_samples, kappa):
            print("Generating using Inverse-Wishart")

            N, T = number_of_assets, number_of_samples

            q_IW = 1 / (1 + 2 * kappa)
            T_IW = int(N / q_IW)

            R = np.random.normal(size=(N, T_IW))
            W = R @ R.T / T
            C_IW = (1 - q_IW) * la.inv(W)

            R_inverse_std_diag_from_C_IW = np.diag(1 / np.sqrt(np.diag(C_IW)))
            C = R_inverse_std_diag_from_C_IW @ C_IW @ R_inverse_std_diag_from_C_IW

            self.mvGaussian(C, np.eye(T))


        def exponentialDecay(self, number_of_assets, number_of_samples, autocorrelation_time):
            print("Generating using Exponential Decay")

            N, T = number_of_assets, number_of_samples

            A = np.array(
                [[np.exp(-np.abs(a - b)/autocorrelation_time) for b in range(T)] for a in range(T)]
            )

            self.mvGaussian(np.eye(N), A)

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
