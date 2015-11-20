"""
Cython implementations of Gaussian process covariance kernel functions
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

from libc.math cimport exp


def isotropic_squared_exponential_kernel(double[:, :] K, double[:, :] X,
                                         double[:] theta, double epsilon=1e-8):
    """ Calculates covariance matrix for isotropic squared exponential kernel.

    Calculate the entries of a Gaussian process covariance matrix given
    input features and kernel function parameters. The (i, j)th element of the
    matrix is calculated as::

    K[i, j] = sigma * exp(-0.5 * sum((X[i, :] - X[j, :)**2) / tau**2)
              + epsilon * (i == j)

    With ``epsilon`` a 'jitter' parameter used to improve numerical stability.

    Parameters
    ----------
    K : ndarray
        Empty array of shape ``(n_data, n_data)`` to write output to.
    X : ndarray
        Array of input features of shape ``(n_data, n_dim)``.
    theta : ndarray
        Two-element array of (log) kernel function parameters with the first
        element being the logarithm of the variance parameter ``sigma`` and
        the second being the logarithm of the length-scale parameter ``tau``.
    epsilon: double
        Positive jitter parameter added to diagonal of covariance matrix for
        numerical stability.
    """
    cdef int i, j
    cdef double sigma = exp(theta[0])
    cdef double tau = exp(theta[1])
    for i in range(K.shape[0]):
        K[i, i] = sigma + epsilon
        for j in range(i):
            K[i, j] = 0.
            for k in range(X.shape[1]):
                K[i, j] += (X[i, k] - X[j, k]) ** 2
            K[i, j] = sigma * exp(- K[i, j] / (2. * tau**2))
            K[j, i] = K[i, j]


def diagonal_squared_exponential_kernel(double[:, :] K, double[:, :] X,
                                        double[:] theta, double epsilon=1e-8):
    """ Calculates covariance matrix for diagonal squared exponential kernel.

    Calculate the entries of a Gaussian process covariance matrix given
    input features and kernel function parameters. The (i, j)th element of the
    matrix is calculated as::

    K[i, j] = sigma * exp(-0.5 * sum((X[i, :] - X[j, :)**2 / tau[:]**2))
              + epsilon * (i == j)

    With `epsilon` a 'jitter' parameter used to improve numerical stability.

    Parameters
    ----------
    K : ndarray
        Empty array of shape ``(n_data, n_data)`` to write output to.
    X : ndarray
        Array of input features of shape ``(n_data, n_dim)``.
    theta : ndarray
        ``(1 + n_dim)`` element array of (log) kernel function parameters.
          * ``theta[0]`` is the logarithm of the variance parameter `sigma`,
          * ``theta[1:]`` is a ``n_dim`` length array of the logarithm of the
            length-scale parameters ``tau[k]`` for each of the input feature
            dimensions.
    epsilon: double
        Positive jitter parameter added to diagonal of covariance matrix for
        numerical stability.
    """
    cdef int i, j, k
    cdef double sigma = exp(theta[0])
    for i in range(K.shape[0]):
        K[i, i] = sigma + epsilon
        for j in range(i):
            K[i, j] = 0.
            for k in range(X.shape[1]):
                K[i, j] += ((X[i, k] - X[j, k]) / exp(theta[k + 1])) ** 2
            K[i, j] = sigma * exp(- K[i, j] / 2.)
            K[j, i] = K[i, j]
