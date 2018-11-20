
from .OutlierNormalizer import OutlierNormalizer
from .StandardNormalizer import StandardNormalizer
from .WinsorizationNormalizer import WinsorizationNormalizer


from weakref import ReferenceType
from numpy import ndarray


class MatrixNormalizerAdapter:

    def __init__(self, outer: ReferenceType):
        self.outer = outer

    def standard(self, positive_required=False, verbose=False):

        normalizer = StandardNormalizer(positive_required)
        self.outer().array = normalizer.normalize(self.outer(), verbose=verbose)

    def winsorization(self, positive_required=False, limits=0.05,verbose=False):

        normalizer = WinsorizationNormalizer(positive_required, limits=limits)
        self.outer().array = normalizer.normalize(self.outer(), verbose=verbose)

    def outlier(self, positive_required=False, verbose=False):

        normalizer = OutlierNormalizer(positive_required)
        self.outer().array = normalizer.normalize(self.outer(), verbose=verbose)

