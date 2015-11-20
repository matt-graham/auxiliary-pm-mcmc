# -*- coding: utf-8 -*-
"""
Standard Markov Chain Monte Carlo updates.
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

import numpy as np
import warnings


def metropolis_step(x_curr, log_f_curr, log_f_func, prng, prop_sampler,
                    prop_scales):
    """ Performs single Metropolis Markov chain update.

    Proposed state update is sampled from proposal sampler given current state
    (with proposal density assumed to be symmetric for Metropolis variant) and
    then proposed update accepted with probability

    .. math::

      p_a = \min(1, \exp(\log f(x_p) - \log f(x_c)))

    where :math:`f(\cdot)` is the (unnormalised) target density.

    Parameters
    ----------
    x_curr : ndarray
        Current Markov chain state.
    log_f_curr : float
        Logarithm of target invariant density for Markov chain evaluated at
        current state.
    log_f_func : function or callable object
        Function which calculates the logarithm of the (potentially
        unnormalised target density) at a specified state. Should have call
        signature::
            log_f = log_f_func(x)
        where ``x`` is the state to evaluate the density at and ``log_f`` the
        calculated log density.
    prng : RandomState
        Pseudo-random number generator object (either an instance of a
        ``numpy`` ``RandomState`` or an object with an equivalent
        interface) used to randomly sample accept decisions in MH accept
        step.
    prop_sampler : function or callable object
        Function which returns a proposed new parameter state drawn from
        proposal distribution given a current parameter state. Should have
        a call signature::
            x_prop = prop_sampler(x_curr, prop_params)
        where ``x_curr`` is the current parameter state vector (as a ndarray)
        which the proposal should be conditioned on, ``prop_params`` is a
        ndarray of parameters for the proposal distribution (e.g. standard
        deviation for Gaussian proposals, may also be ``None`` if no free
        parameters in proposal distribution) and ``x_prop`` is the returned
        random propsal distribution draw, again an ndarray.
    prop_params : ndarray
        Array of values to set the parameters of the proposal distribution to.

    Returns
    -------
    x_next : ndarray
        Markov chain state after performing update - if previous state was
        distributed according to target density this state will be too.
    log_f_next : float
        Logarithm of target density at updated state.
    rejection : boolean
        Whether the proposed update was rejected (True) or accepted (False).
    """
    x_prop = prop_sampler(x_curr, prop_scales)
    log_f_prop = log_f_func(x_prop)
    if prng.uniform() < np.exp(log_f_prop - log_f_curr):
        return x_prop, log_f_prop, False
    else:
        return x_curr, log_f_curr, True


def met_hastings_step(x_curr, log_f_curr, log_f_func, prng, prop_sampler,
                      prop_params, log_prop_density):
    """ Performs single Metropolis-Hastings Markov chain update.

    Proposed state update is sampled from proposal sampler given current state
    and then proposed update accepted with probability

    .. math::

      p_a = \min(1, \exp(\log f(x_p) + \log q(x_c | x_p) -
                         \log f(x_c) - \log q(x_p | x_c)))

    where :math:`f(\cdot)` is the (unnormalised) target density and
    :math:`q(\cdot | \cdot)` is the potentially asymmetric proposal density.

    Parameters
    ----------
    x_curr : ndarray
        Current Markov chain state.
    log_f_curr : float
        Logarithm of target invariant density for Markov chain evaluated at
        current state.
    log_f_func : function or callable object
        Function which calculates the logarithm of the (potentially
        unnormalised target density) at a specified state. Should have call
        signature::
            log_f = log_f_func(x)
        where ``x`` is the state to evaluate the density at and ``log_f`` the
        calculated log density.
    prng : RandomState
        Pseudo-random number generator object (either an instance of a
        ``numpy`` ``RandomState`` or an object with an equivalent
        interface) used to randomly sample accept decisions in MH accept
        step.
    prop_sampler : function or callable object
        Function which returns a proposed new parameter state drawn from
        proposal distribution given a current parameter state. Should have
        a call signature::
            x_prop = prop_sampler(x_curr, prop_params)
        where ``x_curr`` is the current parameter state vector (as a ndarray)
        which the proposal should be conditioned on, ``prop_params`` is a
        ndarray of parameters for the proposal distribution (e.g. standard
        deviation for Gaussian proposals, may also be ``None`` if no free
        parameters in proposal distribution) and ``x_prop`` is the returned
        random propsal distribution draw, again an ndarray.
    prop_params : ndarray
        Array of values to set the parameters of the proposal distribution to.
    log_prop_density : function or callable object
        Function which returns the logarithm of the proposal distribution
        density at a specified state given a current state and set of
        parameters. Should have a call signature::
            log_prop_dens = log_prop_density(x_prop, x_curr, prop_params)
        where ``x_prop`` is the proposed state to evaluate the log density at,
        ``x_curr`` is the state to condition the proposal distribution on,
        ``prop_params`` are the values of any free parameters in the proposal
        distribution density function and ``log_prop_dens`` is the calculated
        log proposal density value.

    Returns
    -------
    x_next : ndarray
        Markov chain state after performing update - if previous state was
        distributed according to target density this state will be too.
    log_f_next : float
        Logarithm of target density at updated state.
    rejection : boolean
        Whether the proposed update was rejected (True) or accepted (False).
    """
    x_prop = prop_sampler(x_curr, prop_params)
    log_f_prop = log_f_func(x_prop)
    log_prop_dens_fwd = log_prop_density(x_prop, x_curr, prop_params)
    log_prop_dens_bwd = log_prop_density(x_curr, x_prop, prop_params)
    accept_prob = np.exp(log_f_prop + log_prop_dens_bwd -
                         log_f_curr - log_prop_dens_fwd)
    if prng.uniform() < accept_prob:
        return x_prop, log_f_prop, False
    else:
        return x_curr, log_f_curr, True


def metropolis_indepedence_step(x_curr, log_f_curr, log_f_func, prng,
                                prop_sampler, prop_params=None,
                                log_prop_density=None):
    """ Performs single Metropolis indepedence sampler Markov chain update.

    Two modes of operation. If ``log_prop_density`` is specified it is assumed
    the target density is of the form

    .. math::

       \pi(x) \propto f(x)

    The proposed state update is then sampled independently from the proposal
    distribution :math:`q(x)` and the proposed update is accepted with
    probability

    .. math::

      p_a = \min(1, \exp(\log f(x_p) + \log q(x_c) -
                         \log f(x_c) - \log q(x_p)))

    where :math:`f(\cdot)` is the (unnormalised) target density and
    :math:`q(\cdot)` is the proposal density.

    Alternatively if ``log_prop_density`` is not specified it is assumed
    the target density is of the form

    .. math::

       \pi(x) \propto f(x) q(x)

    where :math:`q(x)` is a 'prior' density which the ``prop_sampler`` object
    returns a sample from. In this case the acceptance probability has the form

    .. math::

      p_a = \min(1, \exp(\log f(x_p) + \log q(x_p) + \log q(x_c) -
                         \log f(x_c) + \log q(x_c) - \log q(x_p)))

    which simplifies to

    .. math::

      p_a = \min(1, \exp(\log f(x_p) - \log f(x_c)))

    i.e. the :math:`q` terms cancel.

    Parameters
    ----------
    x_curr : ndarray
        Current Markov chain state.
    log_f_curr : float
        One of

        - Logarithm of the (potentially unnormalised) target invariant
          density for Markov chain evaluated at current state if
          ``log_prop_density != None``.
        - Logarithm of 'non-prior' factor in target invariant density
          evaluated at the current state if ``log_prop_density == None``,
          i.e. the factor which the density which ``prop_sampler`` returns a
          sample from is multiplied by in the target density.

    log_f_func : function or callable object
        One of

        - Function which calculates the logarithm of the (potentially
          unnormalised) target invariant density for Markov chain at a
          specified state if  ``log_prop_density != None``.
        - Function which calculates the logarithm of 'non-prior' factor in
          the (potentially unnormalised) target invariant density at a
          specified state if ``log_prop_density == None``.

        Should have call signature::

            log_f = log_f_func(x)

        where ``x`` is the state to evaluate the density at and ``log_f`` the
        calculated log density.
    prng : RandomState
        Pseudo-random number generator object (either an instance of a
        ``numpy`` ``RandomState`` or an object with an equivalent
        interface) used to randomly sample accept decisions in MH accept
        step.
    prop_sampler : function or callable object
        Function which returns a proposed new parameter state drawn
        independently from the proposal distribution. Should have a call
        signature::

            x_prop = prop_sampler(prop_params)

        if ``prop_params`` are specified or::

            x_prop = prop_sampler()

        if ``prop_params == None``.
    prop_params : ndarray
        Array of values to set the parameters of the proposal distribution to.
        May also be set to ``None`` if proposal distribution has no free
        parameters to set.
    log_prop_density : function or callable object or None
        Function which returns the logarithm of the proposal distribution
        density at a specified state and optionally a set of parameters.
        Should have a call signature if ``prop_params != None``::

            log_prop_dens = log_prop_density(x_prop, prop_params)

        or if ``prop_params == None``::

            log_prop_dens = log_prop_density(x_prop)

        where ``x_prop`` is the proposed state to evaluate the log density at,
        and ``log_prop_dens`` is the calculated log proposal density value.
        May also be set to ``None`` if second mode of operation is being used
        as described above.

    Returns
    -------
    x_next : ndarray
        Markov chain state after performing update - if previous state was
        distributed according to target density this state will be too.
    log_f_next : float
        Logarithm of target density at updated state.
    rejection : boolean
        Whether the proposed update was rejected (True) or accepted (False).
    """
    if prop_params:
        x_prop = prop_sampler(prop_params)
    else:
        x_prop = prop_sampler()
    log_f_prop = log_f_func(x_prop)
    if log_prop_density:
        if prop_params:
            log_prop_dens_fwd = log_prop_density(x_prop, prop_params)
            log_prop_dens_bwd = log_prop_density(x_curr, prop_params)
        else:
            log_prop_dens_fwd = log_prop_density(x_prop)
            log_prop_dens_bwd = log_prop_density(x_curr)
        accept_prob = np.exp(log_f_prop + log_prop_dens_bwd -
                             log_f_curr - log_prop_dens_fwd)
    else:
        accept_prob = np.exp(log_f_prop - log_f_curr)
    if prng.uniform() < accept_prob:
        return x_prop, log_f_prop, False
    else:
        return x_curr, log_f_curr, True


class MaximumIterationsExceededError(Exception):
    """ Error raised when iterations of a loop exceeds a predefined limit. """
    pass


def elliptical_slice_step(x_curr, log_f_curr, log_f_func, prng,
                          gaussian_sample, max_slice_iters=1000):
    """ Performs single elliptical slice sampling update.

    Markov chain update for a target density of the form

    .. math::

      \pi(x) \propto N(x | 0, \Sigma) f(x)

    where  :math:`N(x | 0, \Sigma)` represents a zero-mean multivariate
    Gaussian density with covariance matrix :math:`\Sigma` and :math:`f(x)`
    is some non-Gaussian factor in the target density (e.g. a likelihood).

    **Reference:**
    `Elliptical slice sampling`,  Murray, Adams and Mackay (2010)

    Parameters
    ----------
    x_curr : ndarray
        Current Markov chain state.
    log_f_curr : float
        Logarithm of the non-Gaussian target density factor evaluated at
        current state.
    log_f_func : function or callable object
        Function which calculates the logarithm of the non-Gaussian target
        density factor at a specified state. Should have call signature::
            log_f = log_f_func(x)
        where ``x`` is the state to evaluate the density at and ``log_f`` the
        calculated log density.
    prng : RandomState
        Pseudo-random number generator object (either an instance of a
        ``numpy`` ``RandomState`` or an object with an equivalent
        interface).
    gaussian_sample : ndarray
        Independent sample from the Gaussian factor in the target density
        with zero mean and covariance matrix :math:`\Sigma`.
    max_slice_iters : integer
        Maximum number of elliptical slice shrinking iterations to perform
        before terminating and raising an ``MaximumIterationsExceededError``
        exception. This should be set to a relatively large value (e.g. the
        default is 1000) which is significantly larger than the expected number
        of slice shrinking iterations so that this exception is only raised
        when there is some error condition e.g. when there is a bug in the
        implementation of the ``log_f_func`` which would otherwise cause the
        shriking loop to never be terminated.

    Returns
    -------
    x_next : ndarray
        Markov chain state after performing update - if previous state was
        distributed according to target density this state will be too.
    log_f_next : float
        Logarithm of non-Gaussian factor in target density at updated state.

    Raises
    ------
    MaximumIterationsExceededError
        Raised when slice shrinking loop does not terminate within the
        specified limit.
    """
    # draw random log slice height between -infinity and log_f_curr
    log_y = log_f_curr + np.log(prng.uniform())
    # draw first proposed slice angle and use to define intial bracket
    phi = prng.uniform() * 2. * np.pi
    phi_min = phi - 2. * np.pi
    phi_max = phi
    i = 0
    while i < max_slice_iters:
        # calculate proposed state on ellipse defined by Gaussian sample and
        # slice angle and calculate logarithm of non-Gaussian factor
        x_prop = x_curr * np.cos(phi) + gaussian_sample * np.sin(phi)
        log_f_prop = log_f_func(x_prop)
        # check if proposed state on slice if not shrink
        if log_f_prop > log_y:
            return x_prop, log_f_prop
        elif phi < 0:
            phi_min = phi
        elif phi > 0:
            phi_max = phi
        else:
            warnings.warn('Slice collapsed to current value')
            return x_curr, log_f_curr
        # draw new slice angle from updated bracket
        phi = phi_min + prng.uniform() * (phi_max - phi_min)
        i += 1
    raise MaximumIterationsExceededError(
        'Exceed maximum slice iterations: '
        'i={0}, phi_min={1}, phi_max={2}, log_f_prop={3}, log_f_curr={4}'
        .format(i, phi_min, phi_max, log_f_prop, log_f_curr))


def linear_slice_step(x_curr, log_f_curr, log_f_func, slice_width, prng,
                      max_steps_out=0, max_slice_iters=1000):
    """ Performs single linear slice sampling update.

    Performs slice sampling along some line in the target distribution state
    space. This line might be axis-aligned corresponding to sampling along
    only one of the dimensions of the target distribution or some arbitrary
    linear combination of the dimensions.

    The first step in a slice sampling update is to randomly sample a slice
    height between 0 and the (potentially unnormalised) density at the current
    Markov chain state. The set of all the points on the line with a density
    above this slice height value are defined as the current slice and moving
    to a state corresponding to a point drawn uniformly from this slice on
    the line will leave the target distribution invariant. To achieve this
    the first step is to randomly position a bracket of a specified width
    around the current point on the line, optionally stepping out this bracket
    until its ends lie outside the slice. Points are then repeatedly drawn at
    uniform from the current bracket and tested to see if they are in the slice
    (i.e. have a density above the slice height), if they are the current point
    returned otherwise rejected proposed points are used to adaptively shrink
    the slice bracket while maintaining the reversibility of the algorithm.

    **Reference:**
    `Slice sampling`, Neal (2003)

    Parameters
    ----------
    x_curr : ndarray
        Point on line corresponding to current Markov chain state.
    log_f_curr : float
        Logarithm of the potentially unnormalised target density evaluated at
        current state.
    log_f_func : function or callable object
        Function which calculates the logarithm of the potentially unnormalised
        target density at a point on the line. Should have call signature::
            log_f = log_f_func(x)
        where ``x`` is the position on the line to evaluate the density at and
        ``log_f`` the calculated log density.
    slice_width : float
        Initial slice bracket width with bracket of this width being randomly
        positioned around current point.
    prng : RandomState
        Pseudo-random number generator object (either an instance of a
        ``numpy`` ``RandomState`` or an object with an equivalent
        interface).
    max_steps_out : integer
        Maximum number of stepping out iterations to perform (default 0). If
        non-zero then the initial slice bracket  is linearly 'stepped-out' in
        positive and negative directions by ``slice_width`` each time, until
        either the slice bracket ends are outside the slice or the maximum
        number of steps has been reached.
    max_slice_iters : integer
        Maximum number of slice bracket shrinking iterations to perform
        before terminating and raising an ``MaximumIterationsExceededError``
        exception. This should be set to a relatively large value (e.g. the
        default is 1000) which is significantly larger than the expected number
        of slice shrinking iterations so that this exception is only raised
        when there is some error condition e.g. when there is a bug in the
        implementation of the ``log_f_func`` which would otherwise cause the
        shriking loop to never be terminated.

    Returns
    -------
    x_next : ndarray
        Point on line corresponding to new Markov chain state after performing
        update - if previous state was distributed according to target density
        this state will be too.
    log_f_next : float
        Logarithm of target density at updated state.

    Raises
    ------
    MaximumIterationsExceededError
        Raised when slice shrinking loop does not terminate within the
        specified limit.
    """
    # draw random log slice height between -infinity and log_f_curr
    log_y = np.log(prng.uniform()) + log_f_curr
    # randomly set initial slice bracket of specified width w
    x_min = x_curr - slice_width * prng.uniform()
    x_max = x_min + slice_width
    # step out bracket if non-zero maximum steps out
    if max_steps_out > 0:
        # randomly split maximum number of steps between up and down steps
        # to ensure reversibility
        steps_down = np.round(prng.uniform() * max_steps_out)
        steps_up = max_steps_out - steps_down
        s = 0
        while s < steps_down and log_y < log_f_func(x_min):
            x_min -= slice_width
            s += 1
        s = 0
        while s < steps_up and log_y < log_f_func(x_max):
            x_max += slice_width
            s += 1
    i = 0
    while i < max_slice_iters:
        # draw new proposed point randomly on current slice bracket and
        # calculate log density at proposed point
        x_prop = x_min + (x_max - x_min) * prng.uniform()
        log_f_prop = log_f_func(x_prop)
        # check if proposed state on slice if not shrink
        if log_f_prop > log_y:
            return x_prop, log_f_prop
        elif x_prop < x_curr:
            x_min = x_prop
        elif x_prop > x_curr:
            x_max = x_prop
        else:
            warnings.warn('Slice collapsed to current value')
            return x_curr, log_f_curr
        i += 1
    raise MaximumIterationsExceededError(
        'Exceed maximum slice iterations: '
        'i={0}, x_min={1}, x_max={2}, log_f_prop={3}, log_f_curr={4}'
        .format(i, x_min, x_max, log_f_prop, log_f_curr))
