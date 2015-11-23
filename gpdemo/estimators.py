# -*- coding: utf-8 -*-
"""
Estimators for marginal likelihood :math:`p(y | \\theta, X)` in a Gaussian
process classification model (with probit likelihood) for use in
pseudo-marginal MCMC samplers.
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

import numpy as np
import scipy.linalg as la
from scipy.misc import logsumexp
from scipy.special import log_ndtr
from .latent_posterior_approximations import laplace_approximation


class LogMarginalLikelihoodLaplaceEstimator(object):
    """ Log marginal likelihood estimator using Laplace approximation.

    Fits a Gaussian approximation to the posterior on the latent function
    values

    .. math::

      q(f | y, \\theta, X) \\approx \\frac{p(y | f) p(f | \\theta, X)}
                                          {p(y | \\theta, X)}

    using Laplace method and then uses the normalising constant for
    Gaussian approximation as estimate for :math:`p(y | \\theta, X)`.
    This will give a biased but deterministic estimate for the marginal
    likelihood.

    It is assumed :math:`p(y | f)` is a product of probit likelihood terms.
    """

    def __init__(self, X, y, kernel_func):
        """ Log marginal likelihood estimator using Laplace approximation.

        Parameters
        ----------
        X : ndarray
            Array of input features of shape ``(n_data, n_dim)``
        y : ndarray
            Array of binary target values of shape ``(n_data, )``
        kernel_func : function
            Function which calculates a covariance matrix given inputs
            ``(X, theta)`` where `X` is array of input features as above and
            ``theta`` is kernel parameters. The function signature should be
            of the form ``kernel_func(K_out, X, theta)`` where `K_out` is an
            empty array of dimensions ``(n_data, n_data)`` which the covariance
            matrix will be written to.
        """
        self.X = X
        self.y = y
        self.kernel_func = kernel_func
        self._K = np.empty((X.shape[0], X.shape[0]))
        self.n_cubic_ops = 0

    def reset_cubic_op_count(self):
        """ Reset the count of executed ops with order ``n_data**3`` cost. """
        self.n_cubic_ops = 0

    def __call__(self, theta):
        """ Calculate the approximate log marginal likelihood.

        Parameters
        ----------
        theta : ndarray
            Array of kernel function parameters.

        Returns
        -------
        float
            Approximate log marginal likelihood.
        """
        self.kernel_func(self._K, self.X, theta)
        f_post, approx_log_marg_lik, cubic_ops = laplace_approximation(
            self._K, self.y, calc_cov=False, calc_lml=True)
        self.n_cubic_ops += cubic_ops
        return approx_log_marg_lik


class InvalidCovarianceMatrixError(Exception):
    """ Raised when posterior approx. returns a non-PSD covariance matrix. """
    pass


class LogMarginalLikelihoodApproxPosteriorISEstimator(object):
    """ Log marginal likelihood importance sampling from ~posterior estimator.

    Fits a Gaussian approximation to the posterior on the latent function
    values

    .. math::

      q(f | y, \\theta, X) \\approx \\frac{p(y | f) p(f | \\theta, X)}
                                          {p(y | \\theta, X)}

    using provided function and then draws approximate latent function value
    samples from this distribution and uses them to form an importance
    sampling estimate for the marginal likelihood :math:`p(y | \\theta)`.
    The estimate of the marginal likelihood (i.e. the exponential of what is
    returned here) is unbiased.

    It is assumed :math:`p(y | f)` is a product of probit likelihood terms.
    """

    def __init__(self, X, y, kernel_func, post_approx_func):
        """ Log marginal likelihood importance sampling estimator.

        Parameters
        ----------
        X : ndarray
            Array of input features of shape ``(n_data, n_dim)``
        y : ndarray
            Array of binary target values of shape ``(n_data, )``
        kernel_func : function
            Function which calculates a covariance matrix given inputs
            ``(X, theta)`` where ``X`` is array of input features as above and
            ``theta`` is kernel parameters. The function signature should be
            of the form ``kernel_func(K_out, X, theta)`` where ``K_out`` is an
            empty array of dimensions ``(n_data, n_data)`` which the covariance
            matrix will be written to.
        post_approx_func : function
            Function which calculates a Gaussian approximation to the posterior
            :math:`p(f | y, \\theta, X)`. The function signature should be of
            the form ``f_post, C, cubic_ops = post_approx_func(K, y)`` with
            outputs
              * ``f_post``: the approximate posterior mean,
              * ``C``: approximate posterior covariance matrix,
              * ``cubic_ops``:
                the number of operations with order ``n_data**3``
                computational cost executed during calculation of the
                posterior approximation,
            and inputs
              * ``K``: the covariance matrix on the GP latent function prior
              * ``y``: the binary target outputs
        """
        self.X = X
        self.y = y
        self.kernel_func = kernel_func
        self.post_approx_func = post_approx_func
        self._K = np.empty((X.shape[0], X.shape[0]))
        self.n_cubic_ops = 0

    def reset_cubic_op_count(self):
        """ Reset the count of executed ops with order ``n_data**3`` cost. """
        self.n_cubic_ops = 0

    def __call__(self, ns, theta=None, cached_results=None):
        """ Calculate the approximate log marginal likelihood.

        Parameters
        ----------
        ns : ndarray
            Array of independent normal random draws of size
            ``(n_data, n_imp_sample)`` used to generate the ``n_imp_sample``
            samples from the approximate Gaussian posterior used to calculate
            the importance sampling estimate of the marginal log likelihood.
        theta : ndarray
            Array of kernel function parameters.
            Optional - if ``None`` then a value for ``cached_results`` must be
            provided instead.
        cached_results : iterable
            Tuple (or other iterable) of structure
            ``cached_results = (K_chol, C_chol, f_post)``
            where
              * ``K_chol`` is the (lower-triangular) Cholesky decomposition
                of the latent function value GP prior covariance matrix,
              * ``C_chol`` is the (lower-triangular) Cholesky decomposition
                of the approximate posterior covariance matrix
              * ``f_post`` is the approximate posterior mean,
            corresponding to the ``theta`` value which it is required to get
            an estimate for the marginal likelihood. This allows an estimate
            for the marginal likelihood to be computed at a much reduced
            order ``n_data**2`` computational cost for a given ``theta`` if
            ``cached_results`` are stored from a previous call for this
            ``theta`` value.
            Optional - if ``None`` then a value for ``theta`` must be provided
            instead.

        Returns
        -------
        log_p_y_gvn_theta_est : float
            Log marginal likelihood estimate.
        ``(K_chol, C_chol, f_post)`` : (ndarray, ndarray, ndarray)
            Tuple of cached results ``(K_chol, C_chol, f_post)`` as
            described for input parameter ``cached_reults`` above for the
            specified ``theta`` value which can be used to enable more
            efficient calculation of further marginal likelihood estimates for
            this ``theta`` value.

        Raises
        ------
        InvalidCovarianceMatrixError
            If posterior approximation function returns a covariance matrix
            which is not positive semi-definite.
        """
        if theta is None and cached_results is None:
            raise ValueError('One of theta or cached_results must be provided')
        elif cached_results is None:
            # calculate kernel matrix and approximate posterior
            self.kernel_func(self._K, self.X, theta)
            K_chol = la.cholesky(self._K, lower=True)  # cubic op
            f_post, C, cubic_ops = self.post_approx_func(self._K, self.y)
            try:
                C_chol = la.cholesky(C, lower=True)  # cubic op
            except la.LinAlgError:
                e, R = la.eigh(C)
                raise InvalidCovarianceMatrixError(
                    'Posterior covariance matrix not PSD: '
                    'sum of negative eigenvalues {0}'
                    .format(e[e <= 0].sum()))
            # total cubic ops = # in post. apprx + 2 extra chol
            self.n_cubic_ops += cubic_ops + 2
        else:
            # use decompositions of K and C from cached_results plus f_post
            K_chol, C_chol, f_post = cached_results
        n_imp_sample = ns.shape[1]
        # generate samples from latent function approximate posterior
        f_s = f_post[None] + C_chol.dot(ns).T
        # calculate log density of latent function samples under GP prior
        f_s_K_inv_f_s = (la.cho_solve((K_chol, True), f_s.T) * f_s.T).sum(0)
        log_p_f_s_gvn_theta = (-0.5 * f_s_K_inv_f_s -
                               np.log(K_chol.diagonal()).sum())
        # calculate log likelihood of latent function samples given observed y
        log_p_y_gvn_f_s = log_ndtr(f_s * self.y).sum(-1)
        # calculate log density of latent function samples under approximate
        # Gaussian posterior importance sampling distribution
        f_s_zm = f_s - f_post[None]
        f_s_zm_C_inv_f_s_zm = (
            la.cho_solve((C_chol, True), f_s_zm.T) * f_s_zm.T).sum(0)
        log_q_f_s_gvn_y_theta = (-0.5 * f_s_zm_C_inv_f_s_zm -
                                 np.log(C_chol.diagonal()).sum())
        # calculate log marginal likelihood estimate for each importance sample
        log_p_y_gvn_theta_est = (log_p_y_gvn_f_s + log_p_f_s_gvn_theta -
                                 log_q_f_s_gvn_y_theta)
        return (logsumexp(log_p_y_gvn_theta_est) - np.log(n_imp_sample),
                (K_chol, C_chol, f_post))


class LogMarginalLikelihoodPriorMCEstimator(object):
    """ Log marginal likelihood Monte Carlo estimator.

    Samples from the GP latent function prior :math:`p(f | \\theta, X)` and
    uses these samples to form an unbiased Monte Carlo estimate of the marginal
    likelihood :math:`p(y | \\theta)` which can be formulated as the
    expectation over :math:`p(f | \\theta, X)` of :math:`p(y | f)`.

    .. math::

      p(y | \\theta) = \\mathbb{E}_{p(\\cdot | \\theta, X)} [p(y | \\cdot)]

    It is assumed :math:`p(y | f)` is a product of probit likelihood terms.
    """

    def __init__(self, X, y, kernel_func):
        """ Log marginal likelihood importance sampling estimator.

        Parameters
        ----------
        X : ndarray
            Array of input features of shape ``(n_data, n_dim)``
        y : ndarray
            Array of binary target values of shape ``(n_data, )``
        kernel_func : function
            Function which calculates a covariance matrix given inputs
            ``(X, theta)`` where ``X`` is array of input features as above and
            ``theta`` is kernel parameters. The function signature should be
            of the form ``kernel_func(K_out, X, theta)`` where ``K_out`` is an
            empty array of dimensions ``(n_data, n_data)`` which the covariance
            matrix will be written to.
        """
        self.X = X
        self.y = y
        self.kernel_func = kernel_func
        self._K = np.empty((X.shape[0], X.shape[0]))
        self.n_cubic_ops = 0

    def reset_cubic_op_count(self):
        """ Reset the count of executed ops with order ``n_data**3`` cost. """
        self.n_cubic_ops = 0

    def __call__(self, ns, theta=None, K_chol=None):
        """ Calculate the approximate log marginal likelihood.

        Parameters
        ----------
        ns : ndarray
            Array of independent normal random draws of size
            ``(n_data, n_imp_sample)`` used to generate the ``n_imp_sample``
            samples from the approximate Gaussian posterior used to calculate
            the importance sampling estimate of the marginal log likelihood.
        theta : ndarray
            Array of kernel function parameters.
            Optional - if ``None`` then a value for ``K_chol`` must be
            provided instead.
        K_chol : ndarray
            Cached Cholesky decomposition of kernel matrix ``K``. This
            allows an estimate for the marginal likelihood to be computed at
            a much reduced order ``n_data**2`` computational cost for a given
            ``theta`` if ``K_chol`` is stored from a previous call for this
            ``theta`` value.
            Optional - if ``None`` then a value for ``theta`` must be provided
            instead.

        Returns
        -------
        log_p_y_gvn_theta_est : float
            Log marginal likelihood estimate.
        K_chol : ndarray
            Cached Cholesky decomposition of kernel matrix ``K`` for provided
            ``theta`` value.
        """
        if theta is None and K_chol is None:
            raise ValueError('One of theta or K_chol must be provided')
        elif K_chol is None:
            self.kernel_func(self._K, self.X, theta)
            K_chol = la.cholesky(self._K, lower=True)  # cubic op
            self.n_cubic_ops += 1
        f_s = K_chol.dot(ns).T
        log_p_y_gvn_f_s = log_ndtr(f_s * self.y[None]).sum(-1)
        return logsumexp(log_p_y_gvn_f_s) - np.log(ns.shape[1]), K_chol
