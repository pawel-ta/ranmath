
from .RollingWindowSampler import RollingWindowSampler

from weakref import ReferenceType


class MatrixSamplerAdapter:

    def __init__(self, matrix_reference: ReferenceType):
        self.matrix_reference = matrix_reference

    def rw_autocorrelation_eigenvalues(self, sample_size: int, out_of_sample_size: int, verbose=False):

        sampler = RollingWindowSampler(sample_size, out_of_sample_size)
        return sampler.autocorrelation_eigenvalues(self.matrix_reference(), verbose=verbose)

    def rw_autocorrelation_eigenvectors(self, sample_size: int, out_of_sample_size: int, verbose=False):

        sampler = RollingWindowSampler(sample_size, out_of_sample_size)
        return sampler.autocorrelation_eigenvectors(self.matrix_reference(), verbose=verbose)

    def rw_covariance_cubes(self, sample_size: int, out_of_sample_size: int, verbose=False):

        sampler = RollingWindowSampler(sample_size, out_of_sample_size)
        return sampler.covariance_cubes(self.matrix_reference(), verbose=verbose)

    def rw_data_cubes(self, sample_size: int, out_of_sample_size: int, verbose=False):

        sampler = RollingWindowSampler(sample_size, out_of_sample_size)
        return sampler.data_cubes(self.matrix_reference(), verbose=verbose)