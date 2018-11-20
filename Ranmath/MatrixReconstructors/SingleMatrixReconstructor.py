
from .AbstractReconstructor import AbstractReconstructor

import numpy.linalg as la
import numpy as np
from copy import deepcopy


class SingleMatrixReconstructor(AbstractReconstructor):

    def __init__(self, eigenvectors, eigenvalues):
        super().__init__()
        self.__eigenvectors = deepcopy(eigenvectors)
        self.__eigenvalues = deepcopy(eigenvalues)

    def reconstruct(self):
        return self.__eigenvectors @ np.diag(self.__eigenvalues) @ la.inv(self.__eigenvectors)
