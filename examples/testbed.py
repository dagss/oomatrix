import oomatrix as om
import numpy as np
from oomatrix import Matrix, compute, explain

def ndarray(shape, dtype=np.double):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

n = 10

A = om.Matrix(ndarray((n, n)), 'A')
D = om.Matrix(3 * np.ones(n), 'D', diagonal=True)
u = ndarray((n, 1))

expr = A.factor() * D * u

print explain(expr)
#print compute(expr)

