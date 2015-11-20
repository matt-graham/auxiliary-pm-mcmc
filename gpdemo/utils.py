# -*- coding: utf-8 -*-
"""
Helper functions for auxiliary pseudo marginal MCMC Gaussian process
classification experiments.
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

import os
import datetime
import json
import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt


def gamma_log_pdf(x, a, b):
    """ Logarithm of probability density function for Gamma distribution.

    Parameters
    ----------
    x : float
        Value to evaluate density at.
    a : float
        Shape parameter.
    b : float
        Rate parameters.

    Returns
    -------
    float
        Log density.
    """
    return a * np.log(b) - gammaln(a) + (a - 1) * np.log(x) - b * x


def log_gamma_log_pdf(x, a, b):
    """ Logarithm of probability density function for log-Gamma distribution.

    Here log-Gamma distribution denotes the distribution such that the
    exponential of the random variable is Gamma distributed.

    Parameters
    ----------
    x : float
        Value to evaluate density at.
    a : float
        Shape parameter.
    b : float
        Rate parameters.

    Returns
    -------
    float
        Log density.
    """
    return a * np.log(b) - gammaln(a) + a * x - b * np.exp(x)


def adapt_factor_func(b, n_batch):
    """ Calculates adaption factor to use during an adaptive MH run.

    Based on adaption schedule used in code accompanying paper:

    `Pseudo-Marginal Bayesian Inference for Gaussian Processes`,
    Filippone and Girolami (2013)

    Parameters
    ----------
    b : integer
         Index of current batch of updates (each batch of updates being
         used to calculate an average accept rate).
    n_batch : integer
         Total number batches to be used in full adaptive run.

    Returns
    -------
    adapt_factor : double
         Factor to use to scale changes in adapation of proposal parameters.
    """
    return 5. - min(b + 1, n_batch / 5.) / (n_batch / 5.) * 3.9


def normalise_inputs(X):
    """ Normalise input features to have zero-mean and unit standard-deviation.

    Parameters
    ----------
    X : ndarray
        Array of input features of shape ``(n_data, n_dim)``.

    Returns
    -------
    ndarray
        Normalised input features.
    ndarray
        Mean of original input features along each dimension.
    ndarray
        Standard deviation of original input features along each dimension.
    """
    X_mn = X.mean(0)
    X_sd = X.std(0)
    return (X - X_mn[None]) / X_sd[None], X_mn, X_sd


def save_run(output_dir, tag, thetas, n_reject, n_cubic_ops, comp_time,
             run_params):
    """ Save results and parameters of a sampling experiment run.

    Saves parameter state samples and some basic performance metrics to a
    compressed numpy .npz file and simulation run parameters to JSON file
    as key-value pairs. Both save files are saved with a timestamp prefix (the
    same for both) to prevent overwriting previous run outputs.

    Parameters
    ----------
    output_dir : path as string
        Directory to save output files to.
    tag : string
        Descriptive tag to use in filenames to help identifying runs.
    thetas : ndarray
        2D array of state samples with first dimension indexing successive
        samples and the second dimension indexing each state.
    n_reject : integer or typle
        Number of rejected updates in Metropolis(--Hastings) steps, with
        potentially multiple rejection counts being given as a tuple if there
        were several Metropolis(--Hastings) steps in each update.
    n_cubic_ops : integer
        Number of O(N^3) operations (where N is number of data points)
        performed during sampling run.
    comp_time : float
        Wall clock time for sampling run in seconds.
    run_params : dict
        Dictionary of parameters used to specify run.
    """
    time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_')
    results_file = os.path.join(output_dir, time_stamp + tag + '_results.npz')
    params_file = os.path.join(output_dir, time_stamp + tag + '_params.json')
    if hasattr(n_reject, '__len__'):
        perf_stats = np.array([n for n in n_reject] + [n_cubic_ops, comp_time])
    else:
        perf_stats = np.array([n_reject, n_cubic_ops, comp_time])
    np.savez(results_file, thetas=thetas,
             n_reject_n_cubic_ops_comp_time=perf_stats,
             )
    with open(params_file, 'w') as f:
        json.dump(run_params, f, indent=4, sort_keys=True)


def save_adaptive_run(output_dir, tag, adapt_thetas, adapt_prop_scales,
                      adapt_accept_rates, thetas, n_reject, n_cubic_ops,
                      comp_time, run_params):
    """ Save results and parameters of an adaptive sampling experiment run.

    Saves adaptive run results, parameter state samples and some basic
    performance metrics to a compressed numpy .npz file and simulation run
    parameters to JSON file as key-value pairs. Both save files are saved with
    a timestamp prefix (the same for both) to prevent overwriting previous run
    outputs.

    Parameters
    ----------
    output_dir : path as string
        Directory to save output files to.
    tag : string
        Descriptive tag to use in filenames to help identifying runs.
    adapt_thetas : ndarray
        Array of batches of parameter state chains sampled during initial
        adaptive run - of shape (n_batch * batch_size, n_dim)
    adapt_prop_scales : ndarray
        Array of proposal distribution scale parameters at end of successive
        batches in the adaptive run, these being the parameters the
        adaption is using to control the accept rate.
    adapt_accept_rates : ndarray
        Array of average batch accept rates during adaptive run.
    thetas : ndarray
        2D array of state samples with first dimension indexing successive
        samples and the second dimension indexing each state.
    n_reject : integer or typle
        Number of rejected updates in Metropolis(--Hastings) steps, with
        potentially multiple rejection counts being given as a tuple if there
        were several Metropolis(--Hastings) steps in each update.
    n_cubic_ops : integer
        Number of O(N^3) operations (where N is number of data points)
        performed during sampling run.
    comp_time : float
        Wall clock time for sampling run in seconds.
    run_params : dict
        Dictionary of parameters used to specify run.
    """
    time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_')
    results_file = os.path.join(output_dir, time_stamp + tag + '_results.npz')
    params_file = os.path.join(output_dir, time_stamp + tag + '_params.json')
    if hasattr(n_reject, '__len__'):
        perf_stats = np.array([n for n in n_reject] + [n_cubic_ops, comp_time])
    else:
        perf_stats = np.array([n_reject, n_cubic_ops, comp_time])
    np.savez(results_file,
             adapt_thetas=adapt_thetas,
             adapt_prop_scales=adapt_prop_scales,
             adapt_accept_rates=adapt_accept_rates,
             thetas=thetas,
             n_reject_n_cubic_ops_comp_time=perf_stats
             )
    with open(params_file, 'w') as f:
        json.dump(run_params, f, indent=4, sort_keys=True)


def plot_trace(thetas, fig_size=(12, 8)):
    """ Plot a Markov chain parameter state trace for a sampling run.

    Parameters
    ----------
    thetas : ndarray
        2D array with first dimension indexing successive state samples and
        the second dimension being of length 2 and corresponding to the two
        isotropic squared exponential hyperparameters :math:`\log \tau` the
        log length-scale and :math:`\log \sigma` the log variance.
    fig_size : tuple
        Tuple of dimensions (width, height) in inches to set figure size to.

    Returns
    -------
    fig : matplotlib.figure
        Top-level figure object.
    ax1 : matplotlib.axes
        Axes object for log variance plot.
    ax2 : matplotlib.axes
        Axes object for log length-scale plot.
    """
    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_subplot(211)
    ax1.plot(thetas[:, 0])
    ax1.set_xlabel('Number of updates', fontsize=12)
    ax1.set_ylabel(r'$\log\,\sigma$', fontsize=18)
    ax2 = fig.add_subplot(212)
    ax2.plot(thetas[:, 1])
    ax2.set_xlabel('Number of updates', fontsize=12)
    ax2.set_ylabel(r'$\log\,\tau$', fontsize=18)
    return fig, ax1, ax2
