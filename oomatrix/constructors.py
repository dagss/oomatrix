import numpy as np

from .matrix import Matrix
from .selection import RangeSelection

# Misc factories. Many of these should be implemented as separate types
# rather than using diagonal matrices

def diagonal_matrix(array, name=None):
    return Matrix(array, name=name, diagonal=True)

def identity_matrix(n):
    return diagonal_matrix(np.ones(n), 'I')

def zero_matrix(n):
    return diagonal_matrix(np.zeros(n), '0')

def block_diagonal_matrix(matrices):
    matrices = list(matrices)
    i = 0
    n = sum(M.ncols for M in matrices)
    R = 0 # result matrix
    for idx, M in enumerate(matrices):
        assert M.ncols == M.nrows
        # Make projection P so that P.h * M * P puts M on the right place
        # on the diagonal of R
        P = Matrix(RangeSelection(M.nrows, n, (0, M.nrows), (i, i + M.ncols)),
                   'P%d' % idx)
        R += P.h * M * P
        i += M.ncols
    return R

def stack_matrices_horizontally(matrices):
    matrices = list(matrices)
    i = 0
    n = sum(M.ncols for M in matrices)
    R = 0 # result matrix
    for idx, M in enumerate(matrices):
        assert M.ncols == M.nrows
        # Make projection P so that M * P puts M on the right place;
        # R = [M1 M2 M3] = M1 * P1 + M1 * P2 + M3 * P3
        P = Matrix(RangeSelection(M.nrows, n, (0, M.nrows), (i, i + M.ncols)),
                   'P%d' % idx)
        R += M * P
        i += M.ncols
    return R

