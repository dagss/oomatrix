import numpy as np
import sys
from .computer import MatVecComputer, DescriptionWriter, NoopWriter

class Vector(object):
    """
    User-facing handle for symbolic product of matrix and vector.
    """

    def __init__(self, matrix_expression, array, transpose):
        array = np.asarray(array)
        self._expr = matrix_expression
        self._array = array
        assert not transpose
        
    def compute(self, out=None, verbose=False, noop=False, stream=sys.stderr):
        if verbose:
            writer = DescriptionWriter(stream)
        else:
            writer = NoopWriter()
        array = self._array
        out_shape = (self._expr.nrows,) + self._array.shape[1:]
        if out is None:
            writer.putln("Allocating output of shape %s, dtype %s" % (out_shape, self._expr.dtype))
            out = np.empty(out_shape, dtype=self._expr.dtype)
        else:
            if out.shape != out_shape:
                raise ValueError("out has invalid shape")
        MatVecComputer(writer, noop).compute(self._expr, array, out, False)
        return out

    def describe(self, out=None, stream=sys.stderr):
        self.compute(out, verbose=True, noop=True, stream=stream)
        
