
from .AbstractGenerator import AbstractGenerator
import numpy as np
import scipy.linalg as la


class MultivariateGaussianGenerator(AbstractGenerator):

    def __init__(self, C: np.ndarray, A: np.ndarray):
        super().__init__()
        self.__C = C
        self.__A = A

    def generate(self, verbose=False):

        if verbose:
            print("Generating using Multivariate Gaussian")

        N, T = self.__C.shape, self.__A.shape
        if N[0] != N[1] or T[0] != T[1]:
            raise ValueError('C and A should be square matrices')
        N, T = N[0], T[0]

        C_root = la.sqrtm(self.__C).real
        A_root = la.sqrtm(self.__A).real
        random_matrix = np.random.normal(size=(N, T))

        array = C_root @ random_matrix @ A_root

        return array
