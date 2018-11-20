
from .MultivariateGaussianGenerator import MultivariateGaussianGenerator
from .InverseWishartGenerator import InverseWishartGenerator
from .ExponentialDecayGenerator import ExponentialDecayGenerator

from weakref import ReferenceType


class MatrixGeneratorAdapter:

    def __init__(self, outer: ReferenceType):
        self.outer = outer

    def multivariate_gaussian(self, C, A, verbose=False):

        generator = MultivariateGaussianGenerator(C, A)
        self.outer().array = generator.generate(verbose)

    def inverse_wishart(self, number_of_assets, number_of_samples, kappa, verbose=False):

        generator = InverseWishartGenerator(number_of_assets, number_of_samples, kappa)
        self.outer().array = generator.generate(verbose)

    def exponential_decay(self, number_of_assets, number_of_samples, autocorrelation_time, verbose=False):

        generator = ExponentialDecayGenerator(number_of_assets, number_of_samples, autocorrelation_time)
        self.outer().array = generator.generate(verbose)