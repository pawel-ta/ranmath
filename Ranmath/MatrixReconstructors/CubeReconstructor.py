from .AbstractReconstructor import AbstractReconstructor

import numpy.linalg as la
import numpy as np


class CubeReconstructor(AbstractReconstructor):

    def __init__(self, cube_eigenvectors, cube_eigenvalues):
        super().__init__()
        self.__eigenvectors = cube_eigenvectors
        self.__eigenvalues = cube_eigenvalues

    def reconstruct(self):
        result = []
        for i in range(len(self.__eigenvectors)):
            result.append(self.__eigenvectors @ np.diag(self.__eigenvalues) @ la.inv(self.__eigenvectors))
        return np.array(result)
