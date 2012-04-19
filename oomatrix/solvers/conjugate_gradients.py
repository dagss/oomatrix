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
                        norm_order=None, logger=None):
    """
   

    """
    b = np.asarray(b)
    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype)
    if preconditioner is None:
        preconditioner = Matrix(np.ones(b.shape[0], dtype=b.dtype), diagonal=True)

    info = {}

    # Terminology/variable names follow Shewchuk, ch. B3
    #  r - residual
    #  d - preconditioned residual, "P r"
    #  
    # P = inv(M)
    r = b - compute_array(A * x0)

    residual = ndnorm(r, axis=0, ord=norm_order)
    max_residual = np.max(residual)
    min_residual = np.min(residual)
    if logger is not None:
        logger.info('Initial residuals between %e and %e',
                    min_residual, max_residual)

    d = compute_array(A * r)
    
    delta_0 = delta_new = np.sum(r * d, axis=0)

    info['max_residuals'] = max_residuals = [max_residual]
    info['min_residuals'] = min_residuals = [min_residual]
    info['error'] = None

    eps *= residual

    if np.all(residual < eps):
        info['iterations'] = 0
        return (x0, info)

    x = x0
    for k in xrange(maxit):
        q = compute_array(A * d)
        dAd = np.sum(d * q, axis=0)
        if not np.all(np.isfinite(dAd)) or np.any(dAd == 0):
            raise AssertionError()
        alpha = delta_new / dAd
        x += alpha[None, :] * d
        r -= alpha[None, :] * q
        #if k > 0 and k % 50 == 0:
        #    r_est = r
        #    r = b - compute_array(A * x)
        #    logger.info('Recomputing residual')
        #    #logger.info('Recomputing residual, relative error in estimate: %e',
        #    #            np.linalg.norm(r - r_est) / np.linalg.norm(r))
        #    del r_est

        residual = ndnorm(r, axis=0, ord=norm_order)
        max_residual = np.max(residual)
        min_residual = np.min(residual)
        if logger is not None:
            logger.info('Iteration %d: Residual between %e and %e (terminating '
                        'at %e)', min_residual, max_residual, k+1, residual, eps)
        max_residuals.append(max_residual)
        min_residuals.append(min_residual)
        
        if np.all(residual < eps):
            # Before terminating, make sure to recompute the residual
            # exactly, to avoid terminating too early due to numerical errors
            r_est = r
            r = b - compute_array(A * x)
            #logger.info('Recomputing residual, relative error in estimate: %e',
            #            np.linalg.norm(r - r_est) / np.linalg.norm(r))
            residuals = ndnorm(r, axis=0, ord=norm_order)
            if np.all(residuals < eps):
                info['iterations'] = k + 1
                return (x, info)
            else:
                logger.info('Avoided early termination due to recomputing residual')
                
        s = compute_array(preconditioner * r)
        delta_old = delta_new
        delta_new = np.sum(r * s, axis=0)
        beta = delta_new / delta_old
        d = s + beta * d

    err = ConvergenceError("Did not converge in %d iterations" % maxit)
    if raise_error:
        raise err
    else:
        info['iterations'] = maxit
        info['error'] = err
        return (x, info)
