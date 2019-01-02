from .AbstractResolvent import AbstractResolvent
import numpy as np


class InverseWishartResolvent(AbstractResolvent):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute(kappa, x_arr, eta_th=0.005):

        z_arr = np.array([complex(x, - eta_th) for x in x_arr])
        lambda_IW_min = 1 + 1 / kappa - np.sqrt(1 + 2 * kappa) / kappa
        lambda_IW_max = 1 + 1 / kappa + np.sqrt(1 + 2 * kappa) / kappa
        res_IW_arr = kappa * (
            z_arr * (1 + 1 / kappa) - 1 - np.sqrt(z_arr - lambda_IW_min) * np.sqrt(z_arr - lambda_IW_max)) \
            / (z_arr ** 2)

        return res_IW_arr.real, res_IW_arr.imag
