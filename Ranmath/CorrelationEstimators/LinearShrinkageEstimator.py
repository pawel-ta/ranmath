
from .AbstractEstimator import AbstractEstimator
import numpy as np
import scipy.linalg as la
from ..Norms import frobenius_norm_squared


class LinearShrinkageEstimator(AbstractEstimator):

    def __init__(self):
        super().__init__()

    def get_oracle_alpha(self, sample_covariance_cube, C):

        n_iter, N, _ = sample_covariance_cube.shape
        mu_oracle = C.trace() / N
        alpha_squared_oracle = frobenius_norm_squared(C - mu_oracle * np.eye(N))
        delta_squared_oracle_arr = np.array(
            [
                frobenius_norm_squared(sample_covariance_cube[it] - mu_oracle * np.eye(N))
                for it in range(n_iter)
            ]
        )
        alpha_optimal_oracle = alpha_squared_oracle / delta_squared_oracle_arr.mean()
        return alpha_optimal_oracle

    def get_bonafide_alpha(self, R_array, sample_est_eigenvalues_array):

        n_iter, N, T = R_array.shape

        mu_estimated_arr = sample_est_eigenvalues_array.mean(axis=1)

        delta_squared_estimated_arr = (
        (sample_est_eigenvalues_array - mu_estimated_arr.reshape((n_iter, 1))) ** 2).mean(axis=1)

        arr_via_eigval = np.zeros((n_iter, T))
        for it in range(n_iter):
            for a in range(T):
                R_cols = R_array[it, :, a].reshape((N, 1)) @ R_array[it, :, a].reshape((N, 1)).T
                R_cols_eigval, R_cols_eigvec = la.eigh(R_cols)
                arr_via_eigval[it, a] = ((sample_est_eigenvalues_array[it] - R_cols_eigval) ** 2).mean()

        beta_tilde_squared_estimated_arr = arr_via_eigval.mean(axis=1) / T
        beta_squared_estimated_arr = np.minimum(beta_tilde_squared_estimated_arr, delta_squared_estimated_arr)
        alpha_squared_estimated_arr = delta_squared_estimated_arr - beta_squared_estimated_arr
        alpha_optimal_estimated = alpha_squared_estimated_arr.mean() / delta_squared_estimated_arr.mean()

        return alpha_optimal_estimated

    def estimate_eigenvalues(self, sample_est_eigenvalues_array, alpha, verbose=False):
        return alpha * sample_est_eigenvalues_array + (1 - alpha) * np.ones_like(sample_est_eigenvalues_array)