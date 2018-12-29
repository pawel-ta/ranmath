
import numpy as np
import scipy.linalg as la
from collections import namedtuple

from .AbstractSampler import AbstractSampler


class RollingWindowSampler(AbstractSampler):

    def __init__(self, sample_size: int, out_of_sample_size: int):
        super().__init__()
        self.__sample_size = sample_size
        self.__out_of_sample_size = out_of_sample_size

    def autocorrelation_eigenvalues(self, matrix, verbose=False):

        if verbose:
            print("Fetching eigenvalues")

        sample_cube, out_of_sample_cube = self.covariance_cubes(matrix, verbose=verbose)

        sample_eigenvalues, out_of_sample_eigenvalues = [], []

        for matrix in sample_cube:
            sample_eigenvalues.append(la.eigvals(matrix))

        for matrix in out_of_sample_cube:
            out_of_sample_eigenvalues.append(la.eigvals(matrix))

        eigenvalues = namedtuple("Eigenvalues", ["sample_eigenvalues", "out_of_sample_eigenvalues"]) \
            (np.array(sample_eigenvalues), np.array(out_of_sample_eigenvalues))

        return eigenvalues

    def autocorrelation_eigenvectors(self, matrix, verbose=False):

        if verbose:
            print("Fetching eigenvectors")

        sample_cube, out_of_sample_cube = self.covariance_cubes(matrix, verbose=verbose)

        sample_eigenvectors, out_of_sample_eigenvectors = [], []

        for matrix in sample_cube:
            sample_eigenvectors.append(la.eig(matrix)[1])

        for matrix in out_of_sample_cube:
            out_of_sample_eigenvectors.append(la.eig(matrix)[1])

        eigenvectors = namedtuple("Eigenvectors", ["sample_eigenvectors", "out_of_sample_eigenvectors"]) \
            (np.array(sample_eigenvectors), np.array(out_of_sample_eigenvectors))

        return eigenvectors

    def covariance_cubes(self, matrix, verbose=False):

        if verbose:
            print("Fetching covariance cubes")

        window_size = self.__sample_size + self.__out_of_sample_size

        sample_cube, out_of_sample_cube = [], []

        for k in range(len(matrix.array[0]) - window_size + 1):

            sample_border = k + self.__sample_size

            sample = matrix.array[:, k: sample_border]
            out_of_sample = matrix.array[:, sample_border: sample_border + self.__out_of_sample_size]

            T = matrix.array.shape[1]

            sample_cube.append(sample @ sample.T / T)
            out_of_sample_cube.append(out_of_sample @ out_of_sample.T / T)

        covariance_cubes = namedtuple("CovarianceCubes", ['sample_cube', 'out_of_sample_cube']) \
            (np.array(sample_cube), np.array(out_of_sample_cube))

        return covariance_cubes

    def data_cubes(self, matrix, verbose=False):

        if verbose:
            print("Fetching data cubes")

        window_size = self.__sample_size + self.__out_of_sample_size

        sample_cube, out_of_sample_cube = [], []

        for k in range(len(matrix.array[0]) - window_size + 1):

            sample_border = k + self.__sample_size

            sample = matrix.array[:, k: sample_border]
            out_of_sample = matrix.array[:, sample_border: sample_border + self.__out_of_sample_size]

            sample_cube.append(sample)
            out_of_sample_cube.append(out_of_sample)

        covariance_cubes = namedtuple("DataCubes", ['sample_cube', 'out_of_sample_cube']) \
            (np.array(sample_cube), np.array(out_of_sample_cube))

        return covariance_cubes

