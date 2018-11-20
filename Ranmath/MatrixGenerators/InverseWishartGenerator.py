
from .AbstractGenerator import AbstractGenerator
from .MultivariateGaussianGenerator import MultivariateGaussianGenerator
import numpy as np
import scipy.linalg as la


class InverseWishartGenerator(AbstractGenerator):

    def __init__(self, number_of_assets, number_of_samples, kappa):
        super().__init__()
        self.__number_of_assets = number_of_assets
        self.__number_of_samples = number_of_samples
        self.__kappa = kappa

    def generate(self, verbose=False):

        if verbose:
            print("Generating using Inverse-Wishart")

        N, T = self.__number_of_assets, self.__number_of_samples

        q_IW = 1 / (1 + 2 * self.__kappa)
        T_IW = int(N / q_IW)

        R = np.random.normal(size=(N, T_IW))
        W = R @ R.T / T
        C_IW = (1 - q_IW) * la.inv(W)

        R_inverse_std_diag_from_C_IW = np.diag(1 / np.sqrt(np.diag(C_IW)))
        C = R_inverse_std_diag_from_C_IW @ C_IW @ R_inverse_std_diag_from_C_IW

        mv_generator = MultivariateGaussianGenerator(C, np.eye(T))

        return mv_generator.generate(verbose=verbose)