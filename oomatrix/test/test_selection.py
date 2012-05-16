from .common import *
from .. import Matrix, compute_array, selection

def get_vec(nrows, rows_range, cols_range, n=1, order='C'):
    arr = np.zeros((nrows, n), order=order)
    arr[rows_range[0]:rows_range[1], :] = np.arange(cols_range[0], cols_range[1])[:, None]
    return arr

def test_basic():
    def doit(nrows, ncols, rows_range, cols_range):
        assert rows_range[1] - rows_range[0] == cols_range[1] - cols_range[0]
        cols_min, cols_max = cols_range
        rows_min, rows_max = rows_range
        n = 4

        input = get_vec(ncols, cols_range, cols_range, n)
        output_0 = get_vec(nrows, rows_range, cols_range, n)
        
        # Probe select_rows in detail
        P = Matrix(selection.RangeSelection(nrows, ncols, rows_range, cols_range))
        output = compute_array(P * input)
        assert np.all(output_0 == output)
        output = compute_array(input.T * P.h)
        assert np.all(output_0 == output.T)

        # Probe select_columns
        input, output_0 = output_0.T, input.T
        output = compute_array(input * P)
        assert np.all(output_0 == output)
        output = compute_array(P.h * input.T)
        assert np.all(output_0 == output.T)

        # Do transpose

    yield doit, 10, 8, (3, 9), (2, 8)
    yield doit, 10, 10, (0, 10), (0, 10)

