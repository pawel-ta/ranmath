
from .RollingWindowSampler import RollingWindowSampler

from weakref import ReferenceType


class MatrixSamplerAggregate:

    def __init__(self, outer: ReferenceType):
        self.outer = outer

    def rw_autocorrelation_eigenvalues(self, sample_size: int, out_of_sample_size: int, verbose=False):

        sampler = RollingWindowSampler(sample_size, out_of_sample_size)
        return sampler.autocorrelation_eigenvalues(self.outer(), verbose=verbose)

    def rw_autocorrelation_eigenvectors(self, sample_size: int, out_of_sample_size: int, verbose=False):

        sampler = RollingWindowSampler(sample_size, out_of_sample_size)
        return sampler.autocorrelation_eigenvectors(self.outer(), verbose=verbose)

    def rw_covariance_cubes(self, sample_size: int, out_of_sample_size: int, verbose=False):

        sampler = RollingWindowSampler(sample_size, out_of_sample_size)
        return sampler.covariance_cubes(self.outer(), verbose=verbose)