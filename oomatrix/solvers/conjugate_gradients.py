"""Preconditioned conjugate gradients
"""
import numpy as np
from oomatrix import Matrix, compute_array

class ConvergenceError(RuntimeError):
    pass


def ndnorm(arr, axis, ord=None):
    if ord is None:
        # 2-norm
        x = arr * arr
        return np.sum(x, axis=axis)
    elif np.isinf(ord):
        # max(abs(s))
        return np.max(np.abs(arr), axis=axis)
        raise NotImplementedError()
    else:
        raise NotImplementedError('unsupported norm order')

def conjugate_gradients(A, b, preconditioner=None, x0=None,
                        maxit=10**3, eps=10**-8,
                        relative_eps=True,
                        raise_error=True,
                        norm_order=None, logger=None, compiler=None):
    """
   

    """
    b = np.asarray(b)
    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype, order='F')
    if preconditioner is None:
        preconditioner = Matrix(np.ones(b.shape[0], dtype=b.dtype), diagonal=True)

    info = {}

    # Terminology/variable names follow Shewchuk, ch. B3
    #  r - residual
    #  d - preconditioned residual, "P r"
    #  
    # P = inv(M)
    r = b - compute_array(A * x0, compiler=compiler)

    d = compute_array(preconditioner * r, compiler=compiler).copy('F')
    delta_0 = delta_new = np.sum(r * d, axis=0)
    delta_stop = eps**2 * delta_0

    if logger is not None:
        logger.info('Initial residual %e, stopping at %e', np.sqrt(delta_0), np.sqrt(delta_stop))

    info['residuals'] = residuals = [delta_0]
    info['error'] = None

    x = x0
    for k in xrange(maxit):
        logger.info('Iteration %d: Residual %e (terminating at %e)', k,
                    np.sqrt(delta_new), np.sqrt(delta_stop))
        if delta_new < delta_stop:
            info['iterations'] = k
            return (x, info)            
        
        q = compute_array(A * d, compiler=compiler)
        dAd = np.sum(d * q)
        if not np.isfinite(dAd):
            raise AssertionError("conjugate_gradients: A * d yielded inf values")
        if dAd == 0:
            raise AssertionError("conjugate_gradients: A is singular")
        alpha = delta_new / dAd
        x += alpha * d
        r -= alpha * q
        if k > 0 and k % 50 == 0:
            r_est = r
            r = b - compute_array(A * x)
            logger.info('Recomputing residual, relative error in estimate: %e',
                        np.linalg.norm(r - r_est) / np.linalg.norm(r))

        s = compute_array(preconditioner * r, compiler=compiler)
        delta_old = delta_new
        delta_new = np.sum(r * s, axis=0)
        beta = delta_new / delta_old
        d = s + beta * d

        residuals.append(delta_new)

    err = ConvergenceError("Did not converge in %d iterations" % maxit)
    if raise_error:
        raise err
    else:
        info['iterations'] = maxit
        info['error'] = err
        return (x, info)
