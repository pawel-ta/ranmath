
from .MultivariateGaussianGenerator import MultivariateGaussianGenerator
from .InverseWishartGenerator import InverseWishartGenerator
from .ExponentialDecayGenerator import ExponentialDecayGenerator

from weakref import ReferenceType


class MatrixGeneratorAdapter:

    def __init__(self, matrix_reference: ReferenceType):
        self.matrix_reference = matrix_reference
        self.__last_C = None
        self.__last_A = None

    @property
    def last_C(self):
        return self.__last_C

    @property
    def last_A(self):
        return self.__last_A

    def multivariate_gaussian(self, C, A, verbose=False):

        generator = MultivariateGaussianGenerator(C, A)
        self.matrix_reference().array = generator.generate(verbose)
        self.__last_A = generator.last_A
        self.__last_C = generator.last_C

    def inverse_wishart(self, number_of_assets, number_of_samples, kappa, verbose=False):

        generator = InverseWishartGenerator(number_of_assets, number_of_samples, kappa)
        self.matrix_reference().array = generator.generate(verbose)
        self.__last_A = generator.last_A
        self.__last_C = generator.last_C

    def exponential_decay(self, number_of_assets, number_of_samples, autocorrelation_time, verbose=False):

        generator = ExponentialDecayGenerator(number_of_assets, number_of_samples, autocorrelation_time)
        self.matrix_reference().array = generator.generate(verbose)
        self.__last_A = generator.last_A
        self.__last_C = generator.last_C