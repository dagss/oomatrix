from . import symbolic
from .computation import ImpossibleOperationError
from .cost_value import zero_cost

class Decomposition(object):
    pass

class Factor(Decomposition):
    """Computes the matrix ``F`` such that ``self == F * F.h``

    This decomposition only applies to Hermitian/symmetric
    matrices. It is often the same as `cholesky()`, however, the
    resulting matrix does not have to be lower-triangular. In
    particular, sparse matrices will often return ``F == P * L`` where
    ``P`` is a permutation and ``L`` is a lower-triangular factor,
    because that is faster to compute.

    In general there are many ways to factorize a symmetric matrix;
    this method should return one that is as fast as possible and
    numerically stable.

    See also: ``cholesky()``, ``square_root()``.
    """
    name = 'factor'
    factor_count = 1

    @staticmethod
    def create_computation(kind):
        # todo: dispatch on kind
        class DecompositionComputation:
            name='some-factoring-decomposition'
            @staticmethod
            def compute(matrix_impl):
                return matrix_impl.factor()
            @staticmethod
            def cost(matrix_meta):
                return zero_cost # TODO

        return DecompositionComputation

    @staticmethod
    def get_name(kind):
        return '%s.%s' % (kind.__name__, kind.factor.__name__)

    @staticmethod
    def is_supported_by_kind(kind):
        return hasattr(kind, 'factor')
    
def make_matrix_method(decomposition_cls, name=None):
    if name is None:
        name = decomposition_cls.name
    doc = decomposition_cls.__doc__
    
    def method(matrix):
        from .matrix import Matrix
        kind = matrix.get_type()
        #if not decomposition_cls.is_supported_by_kind(kind):
        #    raise ImpossibleOperationError('%s matrices does not support %s'
        #                                   % (kind.name, name))
        return Matrix(symbolic.DecompositionNode(matrix._expr,
                                                 decomposition_cls))

    method.__name__ = name
    method.__doc__ = doc
    return method
