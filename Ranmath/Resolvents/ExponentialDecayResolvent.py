from .AbstractResolvent import AbstractResolvent
import numpy as np
from sympy import coth


class ExponentialDecayResolvent(AbstractResolvent):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute(tau, x_arr, eta_th=0.005):

        z_arr = np.array([complex(x, - eta_th) for x in x_arr])
        g = np.float64(coth(1 / tau))
        lambda_ED_min = g - np.sqrt(g ** 2 - 1)
        lambda_ED_max = g + np.sqrt(g ** 2 - 1)
        res_ED_arr = (1 + 1 / (np.sqrt(z_arr - lambda_ED_min) * np.sqrt(z_arr - lambda_ED_max))) / z_arr

        return res_ED_arr.real, res_ED_arr.imag
