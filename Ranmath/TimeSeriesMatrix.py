
import weakref
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.stats as st
from collections import namedtuple


class TimeSeriesMatrix:

    def __init__(self):
        self.generate = self.__Generators(weakref.ref(self))
        self.normalize = self.__Normalizers(weakref.ref(self))
        self.characteristics = self.__Characteristics(weakref.ref(self))
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

    class __Normalizers:

        def __init__(self, outer):
            self.outer = outer

        def standard(self, positive_required=False):
            print("Standard normalization")

            abs_min_value = abs(self.outer().array.min(axis=0))

            if positive_required:
                processed_array = self.outer().array + abs_min_value
            else:
                processed_array = self.outer().array

            abs_max_value = abs(processed_array.max(axis=0))
            abs_min_value = abs(processed_array.min(axis=0))

            division_factor = []

            for index in range(len(abs_max_value)):
                division_factor.append(max(abs_max_value[index],
                                           abs_min_value[index]))

            self.outer().array = processed_array / np.array(division_factor)


        def outlier(self):
            print("Outlier normalization")

        def winsorization(self):
            print("Winsorization")

            array = []

            for row in self.outer().array:
                array.append(st.mstats.winsorize(row))

            print("HERE")
            print(np.array(array))
            print("HERE")

            self.outer().array = np.array(array)


    class __Generators:

        def __init__(self, outer):
            self.outer = outer

        def multivariate_gaussian(self, C: np.ndarray, A: np.ndarray):
            print("Generating using Multivariate Gaussian")

            N, T = C.shape, A.shape
            if N[0] != N[1] or T[0] != T[1]:
                raise ValueError('C and A should be square matrices')
            N, T = N[0], T[0]

            C_root = la.sqrtm(C).real
            A_root = la.sqrtm(A).real
            random_matrix = np.random.normal(size=(N, T))

            self.outer().array = C_root @ random_matrix @ A_root

        def inverse_wishart(self, number_of_assets, number_of_samples, kappa):
            print("Generating using Inverse-Wishart")

            N, T = number_of_assets, number_of_samples

            q_IW = 1 / (1 + 2 * kappa)
            T_IW = int(N / q_IW)

            R = np.random.normal(size=(N, T_IW))
            W = R @ R.T / T
            C_IW = (1 - q_IW) * la.inv(W)

            R_inverse_std_diag_from_C_IW = np.diag(1 / np.sqrt(np.diag(C_IW)))
            C = R_inverse_std_diag_from_C_IW @ C_IW @ R_inverse_std_diag_from_C_IW

            self.multivariate_gaussian(C, np.eye(T))

        def exponential_decay(self, number_of_assets, number_of_samples, autocorrelation_time):
            print("Generating using Exponential Decay")

            N, T = number_of_assets, number_of_samples

            A = np.array(
                [[np.exp(-np.abs(a - b)/autocorrelation_time) for b in range(T)] for a in range(T)]
            )

            self.multivariate_gaussian(np.eye(N), A)


    class __Characteristics:

        def __init__(self, outer):
            self.outer = outer

        def autocorrelation_eigenvalues(self, sample_size: int, out_of_sample_size: int):
            print("Fetching eigenvalues")

            sample_cube, out_of_sample_cube = self.covariance_cubes(sample_size, out_of_sample_size)

            sample_eigenvalues, out_of_sample_eigenvalues = [], []

            for matrix in sample_cube:
                sample_eigenvalues.append(la.eigvals(matrix))

            for matrix in out_of_sample_cube:
                out_of_sample_eigenvalues.append(la.eigvals(matrix))

            eigenvalues = namedtuple("Eigenvalues", ["sample_eigenvalues", "out_of_sample_eigenvalues"])\
                (np.array(sample_eigenvalues), np.array(out_of_sample_eigenvalues))

            return eigenvalues

        def autocorrelation_eigenvectors(self, sample_size: int, out_of_sample_size: int):
            print("Fetching eigenvectors")

            sample_cube, out_of_sample_cube = self.covariance_cubes(sample_size, out_of_sample_size)

            sample_eigenvectors, out_of_sample_eigenvectors = [], []

            for matrix in sample_cube:
                sample_eigenvectors.append(la.eig(matrix)[1])

            for matrix in out_of_sample_cube:
                out_of_sample_eigenvectors.append(la.eig(matrix)[1])

            eigenvectors = namedtuple("Eigenvectors", ["sample_eigenvectors", "out_of_sample_eigenvectors"]) \
                (np.array(sample_eigenvectors), np.array(out_of_sample_eigenvectors))

            return eigenvectors

        def covariance_cubes(self, sample_size: int, out_of_sample_size: int):
            print("Fetching covariance cubes")

            window_size = sample_size + out_of_sample_size

            sample_cube, out_of_sample_cube = [], []

            for k in range(len(self.outer().array[0]) - window_size + 1):

                sample_border = k + sample_size

                sample = self.outer().array[:, k: sample_border]
                out_of_sample = self.outer().array[:, sample_border: sample_border + out_of_sample_size]

                sample_cube.append(sample @ sample.T)
                out_of_sample_cube.append(out_of_sample @ out_of_sample.T)

            covariance_cubes = namedtuple("CovarianceCubes", ['sample_cube', 'out_of_sample_cube'])\
                (np.array(sample_cube), np.array(out_of_sample_cube))

            return covariance_cubes
