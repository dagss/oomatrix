import numpy as np

from .matrix import Matrix
from .selection import RangeSelection

# Misc factories. Many of these should be implemented as separate types
# rather than using diagonal matrices

def as_matrix(array, name=None):
    return Matrix(array, name=name)

def diagonal_matrix(array, name=None):
    return Matrix(array, name=name, diagonal=True)

def identity_matrix(n):
    return diagonal_matrix(np.ones(n), 'I')

def zero_matrix(n):
    return diagonal_matrix(np.zeros(n), '0')

def block_diagonal_matrix(matrices):
    matrices = list(matrices)
    i = j = 0
    n = sum(M.nrows for M in matrices)
    m = sum(M.ncols for M in matrices)
    R = 0 # result matrix
    for idx, M in enumerate(matrices):
        # Make projection P, Pp so that P.h * M * Pp puts M on the right place
        # on the diagonal of R
        Pp = Matrix(RangeSelection(M.ncols, m, (0, M.ncols), (i, i + M.ncols)),
                    'Pp%d' % idx)
        P = Matrix(RangeSelection(M.nrows, n, (0, M.nrows), (j, j + M.nrows)),
                   'P%d' % idx)
        R += P.h * M * Pp
        i += M.ncols
        j += M.nrows
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

def block_matrix(matrices):
    n = sum(M.nrows for M in matrices[:, 0])
    m = sum(M.ncols for M in matrices[0, :])

    i = 0
    P_list = []
    for im, M in enumerate(matrices[:, 0]):
        P_list.append(Matrix(RangeSelection(M.nrows, n, (0, M.nrows), (i, i + M.nrows)),
                             'P_%d' % im))
        i += M.nrows

    i = 0
    Pp_list = []
    for im, M in enumerate(matrices[0, :]):
        Pp_list.append(Matrix(RangeSelection(M.ncols, m, (0, M.ncols), (i, i + M.ncols)),
                              'Pp_%d' % im))
        i += M.ncols

    
    R = 0 # result matrix
    for im in range(matrices.shape[0]):
        for jm in range(matrices.shape[1]):
            # Make projection P, Pp so that P.h * M * Pp puts M on the right place
            # on the diagonal of R
            R += P_list[im].h * matrices[im, jm] * Pp_list[jm]
    return R
    
