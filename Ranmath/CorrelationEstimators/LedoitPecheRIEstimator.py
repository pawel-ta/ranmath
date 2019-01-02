
from .AbstractEstimator import AbstractEstimator
from Ranmath.Resolvents import SimulatedEigenvaluesResolvent as resolvent
import numpy as np


class LedoitPecheRIEstimator(AbstractEstimator):

    def __init__(self):
        super().__init__()

    def get_optimal_q(self, number_of_assets, number_of_samples):
        return number_of_assets / number_of_samples

    def get_optimal_eta(self, number_of_assets):
        return number_of_assets ** (-1/2)

    def estimate_eigenvalues(self, sample_est_eigenvalues_array, q, eta=0.005, verbose=False):

        h_arr, r_arr = resolvent.compute_array(sample_est_eigenvalues_array, sample_est_eigenvalues_array, eta)

        rie_eigenvalues = sample_est_eigenvalues_array / (
            (q * sample_est_eigenvalues_array * h_arr - q + 1) ** 2 + (q * sample_est_eigenvalues_array * r_arr) ** 2)

        return rie_eigenvalues
