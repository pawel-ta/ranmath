from .AbstractEstimator import AbstractEstimator
from Ranmath.Resolvents import SimulatedEigenvaluesResolvent as resolvent
import numpy as np


class OracleEstimator(AbstractEstimator):
    def __init__(self):
        super().__init__()

    def estimate_eigenvalues(self, sample_est_eigenvectors_array, C):

        N, _ = sample_est_eigenvectors_array.shape

        return np.array(
                    [
                        sample_est_eigenvectors_array[:, i] @ C @ sample_est_eigenvectors_array[:, i]
                        for i in range( N )
                    ]
                )

