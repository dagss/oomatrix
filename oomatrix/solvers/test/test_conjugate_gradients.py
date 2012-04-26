from oomatrix.test.common import *
from oomatrix import Matrix, compute_array
from ..conjugate_gradients import conjugate_gradients, ndnorm

from matplotlib import pyplot as plt

def test_shewchuck():
    # Example from Shewchuck, converges in 2 iterations
    A = Matrix([[3, 2],[2, 6]])
    b = np.array([[2, 2, 1],
                  [-8, 8, -4]], dtype=np.float64)
    x, info = conjugate_gradients(A, b, maxit=200, eps=10**-8)
    bhat = compute_array(A * x)
    assert np.all(ndnorm(bhat - b, axis=0) / ndnorm(b, axis=0) < 1e-8)
    assert info['iterations'] == 2



    ## Try a bigger matrix; -1 on the first off-diagonal and
    ## increasing sequence on diagonal.
    ## >>> A = _tridiag(100, -1, 2, -1).astype(np.float)
    ## >>> A += np.diag(np.r_[0:100])
    ## >>> np.all(np.linalg.eig(A) > 0) # positive-definite?
    ## True
    ## >>> b = np.r_[10:110].astype(np.float)

    ## Normal:
    
    ## >>> x, k = CG(A, b, maxit=200)
    ## >>> np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b) <= 10**-8
    ## True
    ## >>> k
    ## 49

    ## Be happy with lower convergence:

    ## >>> x, k = CG(A, b, eps=1/10)
    ## >>> np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b) < 1/10
    ## True
    ## >>> k
    ## 3

    ## Use a diagonal preconditioner:

    ## >>> def precond(x): return x / (2+np.r_[0:100])
    ## >>> x, k = CG(A, b, precond=precond)
    ## >>> np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b) <= 10**-8
    ## True
    ## >>> k
    ## 10
