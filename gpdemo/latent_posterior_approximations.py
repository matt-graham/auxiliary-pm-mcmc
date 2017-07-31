# -*- coding: utf-8 -*-
"""
Functions for calculating deterministic approximations to posterior on
latent function values :math:`p(f | y, \\theta, X)` in Gaussian process
classification model with probit likelihood.
"""

import numpy as np
import scipy.linalg as la
from scipy.special import log_ndtr


class MaximumIterationsExceededError(Exception):
    """ Raised when Newton's method fails to converge by iteration limit. """
    pass


def laplace_approximation(K, y, calc_cov=True, calc_lml=False, diff_f_tol=1e-4,
                          max_iters=1000):
    """ Calculates Gaussian approximation to latent function posterior.

    Uses Laplace's method to fit Gaussian approximation to latent function
    posterior :math:`p(f | y, \\theta, X)`, finding mode of the true
    posterior (which is guaranteed to be convex for the probit likelihood) and
    approximating the distribution as a Gaussian centred at this mode with
    covariance matrix equal to the negative Hessian of the true posterior at
    the mode.

    The mode is found using a Newton's method iteration using a numerically
    stable parameterisation based on the pseudo-code in chapter 3 of
    `Gaussian Processes for Machine Learning`, Rasmussen and Williams (2006).
    It is assumed :math:`p(y | f)` is a product of probit likelihood
    terms.

    Parameters
    ----------
    K : ndarray
        Covariance matrix for the Gaussian process prior on the latent function
        values :math:`p(f | \\theta, X)`, shape ``(n_data, n_data)``.
    y : ndarray
        Array of binary target values, shape ``(n_data, )``.
    calc_cov : boolean
        Flag specifiying whether to calculate and return covariance matrix
        C of Gaussian posterior approximation as well as mode.
    calc_lml : boolean
        Flag specifiying whether to calculate and return the approximate
        log marginal likelihood :math:`\\log p(y | \\theta)` calculated from
        the normalising constant for Gaussian approximation.
    diff_f_tol : float
        Convergence tolerance for the Newton's method iteration - iteration
        will be halted when the change in mean of the elementwise square
        differences in the mode estimates between two iterations is less than
        this value.
    max_iters : integer
        Maximum number of iterations to run before abandoning and raising
        an exception.

    Returns
    -------
    f_post : ndarray
        Mean of Gaussian approximation corresponding to mode of true posterior.
    C : ndarray (only if ``calc_cov == True``)
        Covariance matrix of Gaussian approximation.
    lml : float (only if ``calc_lml == True``)
         Approximate log marginal likelihood, :math:`\\log p(y | \\theta)`.
    n_cubic_ops : integer
         Number of order ``n_data**3`` operations executed while fitting the
         approximate posterior.

    Raises
    ------
    MaximumIterationsExceededError
        If Newton's method fails to converge by iteration limit.
    ------

    """
    f = np.zeros(y.shape[0])
    converged = False
    i = 0
    # use Newton's method to find mode
    while not converged and i < max_iters:
        v = np.exp(-0.5 * f**2 - log_ndtr(y * f) - 0.5 * np.log(2 * np.pi))
        grad = v * y
        W_diag = v**2 + grad * f
        W_diag_sqrt = W_diag**0.5
        W_sqrt_K = (W_diag_sqrt * K).T
        B = np.eye(y.shape[0]) + W_sqrt_K * W_diag_sqrt
        L = la.cholesky(B, lower=True)  # cubic op
        b = W_diag * f + grad
        a = b - (W_diag_sqrt * la.cho_solve((L, True), W_sqrt_K.dot(b)))
        f_ = K.dot(a)
        diff = np.mean((f_ - f)**2)
        converged = diff < diff_f_tol
        f = f_
        i += 1
    if not converged:
        raise MaximumIterationsExceededError(
            'Failed to converge in {0} iterations'.format(i))
    if calc_lml:
        # calculate approximate log marginal likelihood
        approx_log_marg_lik = (-0.5 * a.dot(f) + log_ndtr(y * f).sum() -
                               np.log(L.diagonal()).sum())
    if calc_cov:
        # calculate covariance matrix using negative Hessian at mode
        # technically should possibly recalculate B / L here for final
        # update to f, however as change in f negligible, ignore
        B_inv_W_sqrt_K = la.cho_solve((L, True), W_sqrt_K)
        C = K - W_sqrt_K.T.dot(B_inv_W_sqrt_K)  # cubic op
    if calc_cov and calc_lml:
        # i + 1 is total number of O(N**3) ops
        return f, C, approx_log_marg_lik, i + 1
    elif calc_lml:
        # i is total number of O(N**3) ops
        return f, approx_log_marg_lik, i
    elif calc_cov:
        # i + 1 is total number of O(N**3) ops
        return f, C, i + 1
    else:
        # i is total number of O(N**3) ops
        return f, i
