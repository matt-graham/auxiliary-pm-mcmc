# -*- coding: utf-8 -*-
"""
Auxiliary Pseudo-Marginal Markov chain Monte Carlo samplers
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

import numpy as np
import mcmc_updates as mcmc


class BaseAdaptiveMHSampler(object):
    """ Base class for adaptive Metropolis Hastings samplers.

    Implements a basic adaptive MH scheme which tunes scale parameters of the
    MH proposal distributions to achieve an acceptance rate in some target
    range.

    A derived class must implement a ``get_samples`` method with signature::
        thetas, n_reject = get_samples(self, theta_init, n_sample)
    which returns the sampled states ``thetas`` and number of rejections
    ``n_reject`` made during a series of ``n_sample`` iterations of a
    MCMC update dynamic which includes (or solely consists of) a MH update
    step with proposal distribution parameterised by the ``proposal_scales``
    attribute of this class as scale parameters.
    """

    def __init__(self, prop_scales):
        """ Base class for adaptive Metropolis Hastings samplers.

        Parameters
        ----------
        prop_scales : ndarray
            Array of values to initialise proposal distribution scale
            parameters to.
        """
        self.prop_scales = prop_scales

    def get_samples(self, theta_init, n_sample):
        """ Perform a series of Markov chain updates.

        Parameters
        ----------
        theta_init : ndarray
            State to initialise chain at, with shape ``(n_dim, )``.
        n_sample : integer
            Number of Markov chain updates to perform and so state samples to
            return.

        Returns
        -------
        thetas : ndarray
            Two dimensional array of sampled chain states with shape
            ``(n_sample, n_dim)``.
        n_reject : integer or iterable
             For a Markov chain in which each state update contains only one
             Metropolis(-Hastings) accept step this is the number of rejected
             proposed updates during the ``n_sample`` updates. If each update
             contains multiple Metropolis(-Hastings) accept steps this is an
             iterable with each element corresponding to the rejection count
             for a particular accept step in the order they are performed in
             the overall update.
        """
        raise NotImplementedError()

    def adaptive_run(self, theta_init, batch_size, n_batch,
                     low_acc_thr, upp_acc_thr, adapt_factor_func,
                     print_details=False, reject_count_index=-1,):
        """ Run MH Markov chain with proposal tuning to adapt acceptance rate.

        Performs batches of MH Markov chain updates, after each batch
        calculating an estimate of the current acceptance rate from the
        number of rejections in the last batch and is this falls outside
        some specified range, adjusting the MH proposal distribution
        scale parameters by some multiplicate or divisive adaption factor
        calculated as a function of the current batch number and overall
        number of batches.

        Parameters
        ----------
        theta_init : ndarray
            State to start running chain from.
        batch_size : integer
            Number of samples (Markov chain updates) to compute for each batch.
        n_batch : integer
            Number of batches of updates (and so adaptions) to do in total.
        low_acc_thr : float
            Lower acceptance rate threshold, a batch estimated acceptance rate
            less than this will cause the proposal distribution scales to be
            divided by ``adapt_factor_func(b, n_batch)`` where ``b`` is the
            current batch number.
        upp_acc_thr : float
            Upper acceptance rate threshold, a batch estimated acceptance rate
            more than this will cause the proposal distribution scales to be
            multiplied by ``adapt_factor_func(b, n_batch)`` where ``b`` is the
            current batch number.
        adapt_factor_func : function or callable object
            Function which determines the factor by which the proposal
            distribution scale parameters are adjusted after each batch
            (if acceptance rate outside required interval). Function should
            have a signature of the form::
                adapt_factor = adapt_factor_func(b, n_batch)
            where ``adapt_factor`` is a scalar floating-point value used to
            multiply / divide the proposal widths, ``b`` is the current batch
            number and ``n_batch`` is the total number of batches to be used.
        print_details : boolean
            Whether to print accept rate and adaption factor for each batch
            to standard out during a run.
        reject_count_index : integer
            Optional argument specifying index of which rejection count to use
            as adaption signal when ``get_samples`` method returns an iterable
            of rejection counts as its second argument, for example when
            each overall Markov chain update between successive sample is
            composed of several Metropolis(-Hastings) steps each with their
            own possibility of rejecting. The index specified should
            correspond to the rejection count of the MH step of which the
            proposal distribution is parameterised by the ``prop_scales``
            parameters for the adaptive run to make any sense.

        Returns
        -------
        thetas : ndarray
            Array of states sampled during all batches of adaptive run, of
            shape ``(n_batch * n_size, n_dim)``.
        prop_scales : ndarray
            Array of proposal scales after each batch of adaptive run, of
            shape ``(n_batch, n_dim)``.
        accept_rates : ndaray
            Array of acceptance rates for each batch of adaptive run, of shape
            ``(n_batch, )``.
        """
        thetas = np.empty((n_batch * batch_size, theta_init.shape[0]))
        prop_scales = np.empty((n_batch, self.prop_scales.shape[0]))
        accept_rates = np.empty(n_batch)
        for b in range(n_batch):
            thetas[b*batch_size:(b+1)*batch_size], n_reject = (
                self.get_samples(theta_init, batch_size))
            # if multiple rejection counts present e.g. from multiple
            # Metropolis(-Hastings) accept steps during one overall update
            # for different parts of state, adapt only using last
            if hasattr(n_reject, '__len__') and reject_count_index:
                n_reject = n_reject[reject_count_index]
            accept_rates[b] = 1. - (n_reject * 1. / batch_size)
            theta_init = thetas[(b + 1) * batch_size - 1]
            adapt_factor = adapt_factor_func(b, n_batch)
            if accept_rates[b] < low_acc_thr:
                self.prop_scales /= adapt_factor
            elif accept_rates[b] > upp_acc_thr:
                self.prop_scales *= adapt_factor
            prop_scales[b] = self.prop_scales
            if print_details:
                print('Batch {0}: accept rate {1}, adapt factor {2}'
                      .format(b + 1, accept_rates[b], adapt_factor))
        return thetas, prop_scales, accept_rates


class PMMHSampler(BaseAdaptiveMHSampler):
    """ Pseudo-marginal Metropolis Hastings sampler.

    Markov chain Monte Carlo sampler which uses pseudo-marginal Metropolis
    Hastings updates. In the pseudo-marginal framework only an unbiased
    noisy estimate of the (unnormalised) target density is available.
    """

    def __init__(self, log_f_estimator, log_prop_density, prop_sampler,
                 prop_scales, prng):
        """ Pseudo-Marginal Metropolis Hastings sampler.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target distribution given current parameter state. Should
            have a call signature::
                log_f_est = log_f_estimator(theta)
            where ``theta`` is state vector (as ndarray) to evaluate density
            at and log_f_est is the returned double log-density estimate.
        log_prop_density : function or callable object or None
            Function returning logarithm of parameter update proposal density
            at a given proposed parameter state given the current parameter
            state. Should have a call signature::
                log_prop_dens = log_prop_density(theta_prop, theta_curr)
            where ``theta_prop`` is proposed parameter state to evaluate the
            log proposal density at, ``theta_curr`` is the parameter state to
            condition the proposal density on and ``log_prop_dens`` is the
            returned log proposal density value. Alternatively ``None`` may
            be passed which indicates a symmetric proposal density in which
            case a Metropolis update will be made.
        prop_sampler : function or callable object
            Function which returns a proposed new parameter state drawn from
            proposal distribution given a current parameter state. Should have
            a call signature::
                theta_prop = prop_sampler(theta_curr, prop_scales)
            where ``theta_curr`` is the current parameter state vector (as a
            ndarray) which the proposal should be conditioned on,
            ``prop_scales`` is a ndarray of scale parameters for the proposal
            distribution (e.g. standard deviation for Gaussian proposals) and
            ``theta_prop`` is the returned random propsal distribution draw,
            again an ndarray.
        prop_scales : ndarray
            Array of values to initialise the scale parameters of the state
            proposal distribution to. If an initial adaptive run is performed
            by calling ``adaptive_run``, these parameters will be tuned to
            try to achieve an average accept rate in some prescribed interval.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        """
        super(PMMHSampler, self).__init__(prop_scales)
        self.log_f_estimator = log_f_estimator
        if log_prop_density is None:
            self.do_metropolis_update = True
        else:
            self.do_metropolis_update = False
            self.log_prop_density = log_prop_density
        self.prop_sampler = prop_sampler
        self.prng = prng

    def get_samples(self, theta_init, n_sample):
        """ Perform a series of Markov chain updates.

        Parameters
        ----------
        theta_init : ndarray
            State to initialise chain at, with shape ``(n_dim, )``.
        n_sample : integer
            Number of Markov chain updates to perform and so state samples to
            return.

        Returns
        -------
        thetas : ndarray
            Two dimensional array of sampled chain states with shape
            ``(n_sample, n_dim)``.
        n_reject : integer
             The number of rejected proposed updates during the ``n_sample``
             updates.
        """
        if hasattr(theta_init, 'shape'):
            thetas = np.empty((n_sample, theta_init.shape[0]))
        else:
            thetas = np.empty(n_sample)
        thetas[0] = theta_init
        log_f_est_curr = self.log_f_estimator(theta_init)
        n_reject = 0
        for s in range(1, n_sample):
            if self.do_metropolis_update:
                thetas[s], log_f_est_curr, rejection = mcmc.metropolis_step(
                    thetas[s-1], log_f_est_curr, self.log_f_estimator,
                    self.prng, self.prop_sampler, self.prop_scales)
            else:
                thetas[s], log_f_est_curr, rejection = mcmc.met_hastings_step(
                    thetas[s-1], log_f_est_curr, self.log_f_estimator,
                    self.prng, self.prop_sampler, self.prop_scales,
                    self.log_prop_density)
            if rejection:
                n_reject += 1
        return thetas, n_reject


class APMMetIndPlusMHSampler(BaseAdaptiveMHSampler):
    """ Auxiliary pseudo-marginal MI + MH sampler.

    Sampler in the auxiliary pseudo-marginal MCMC framework which uses
    Metropolis independence updates for the random draws and Metropolis--
    Hastings updates for the parameter state.
    """

    def __init__(self, log_f_estimator, log_prop_density, prop_sampler,
                 prop_scales, u_sampler, prng):
        """ Auxiliary Pseudo-marginal MI + MH sampler.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target distribution given current parameter state and
            random draws. Should have a call signature::
                log_f_est, cached_res_out =
                    log_f_estimator(u, theta, [, cached_res_in])
            where ``u`` is the vector of auxiliary random draws used in the
            density estimator, ``theta`` is the state vector (as ndarray) to
            estimate the density at, ``cached_res_in`` is an optional input
            which can be provided if cached intermediate results
            deterministically calculated from the ``theta`` which it is wished
            to estimate the density at have been stored from a previous call,
            potentially speeding subsequent estimates, ``log_f_est`` is the
            calculated log-density estimate and ``cached_res_out`` are
            intermediate cached results determinstically calculated from
            the specified ``theta`` which can be used in subsequent calls to
            potentially speed further estimates of the log density for this
            ``theta`` value (if ``cached_res_in`` was specified then
            ``cached_res_out == cached_res_in``).
        log_prop_density : function or callable object or None
            Function returning logarithm of parameter update proposal density
            at a given proposed parameter state given the current parameter
            state. Should have a call signature::
                log_prop_dens = log_prop_density(theta_prop, theta_curr)
            where ``theta_prop`` is proposed parameter state to evaluate the
            log proposal density at, ``theta_curr`` is the parameter state to
            condition the proposal density on and ``log_prop_dens`` is the
            returned log proposal density value. Alternatively ``None`` may
            be passed which indicates a symmetric proposal density in which
            case a Metropolis update will be made.
        prop_sampler : function or callable object
            Function which returns a proposed new parameter state drawn from
            proposal distribution given a current parameter state. Should have
            a call signature::
                theta_prop = prop_sampler(theta_curr, prop_scales)
            where ``theta_curr`` is the current parameter state vector (as a
            ndarray) which the proposal should be conditioned on,
            ``prop_scales`` is a ndarray of scale parameters for the proposal
            distribution (e.g. standard deviation for Gaussian proposals) and
            ``theta_prop`` is the returned random propsal distribution draw,
            again an ndarray.
        prop_scales : ndarray
            Array of values to initialise the scale parameters of the state
            proposal distribution to. If an initial adaptive run is performed
            by calling ``adaptive_run``, these parameters will be tuned to
            try to achieve an average accept rate in some prescribed interval.
        u_sampler : function or callable object
            Function which returns an independent sample from the 'prior'
            distribution on the random draws :math:`q(u)`.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        """
        super(APMMetIndPlusMHSampler, self).__init__(prop_scales)
        self.log_f_estimator = log_f_estimator
        if log_prop_density is None:
            self.do_metropolis_update = True
        else:
            self.do_metropolis_update = False
            self.log_prop_density = log_prop_density
        self.prop_sampler = prop_sampler
        self.prop_scales = prop_scales
        self.u_sampler = u_sampler
        self.prng = prng

    def get_samples(self, theta_init, n_sample, u_init=None):
        """ Perform a series of Markov chain updates.

        Parameters
        ----------
        theta_init : ndarray
            State to initialise parameters at, with shape ``(n_dim, )``.
        n_sample : integer
            Number of Markov chain updates to perform and so state samples to
            return.
        u_init : ndarray
            State to initialise random draws at. Optional, if not specified
            will be sampled from base density.

        Returns
        -------
        thetas : ndarray
            Two dimensional array of sampled chain states with shape
            ``(n_sample, n_dim)``.
        (n_reject_1, n_reject_2) : tuple
             The number of rejected proposed updates during the ``n_sample``
             updates, in the acceptance step for the random draw variable
             given the current state parameter (``n_reject_1``) and in the
             acceptance step for the state parameter given the current random
             draws (``n_reject_2``).
        """
        if hasattr(theta_init, 'shape'):
            thetas = np.empty((n_sample, theta_init.shape[0]))
        else:
            thetas = np.empty(n_sample)
        thetas[0] = theta_init
        u = u_init if not u_init is None else self.u_sampler()
        log_f_est_curr, cached_res_curr = self.log_f_estimator(u, theta_init)
        n_reject_1 = 0
        n_reject_2 = 0
        for s in range(1, n_sample):
            ## Update u keeping theta fixed using MI
            # As for this update only u will be changed, cached results
            # for current theta calculated in previous log_f_estimator call
            # can be reused, hence pass these values to estimator (with no
            # theta value being needed in this case) and use only first
            # return value (as second will be equal to cached_res_curr)
            log_f_func_1 = lambda v: (
                self.log_f_estimator(v, thetas[s-1], cached_res_curr)[0])
            u, log_f_est_curr, rejection = mcmc.metropolis_indepedence_step(
                u, log_f_est_curr, log_f_func_1, self.prng, self.u_sampler)
            if rejection:
                n_reject_1 += 1
            ## Update theta keeping u fixed using MH

            def log_f_func_2(theta):
                # save cached results from estimator evaluation for proposed
                # theta update so this can be saved to be used in
                # final call of log_f_func in slice sampling routine will
                # always be accepted update so cached results will be correct
                log_f_est, self._cached_res_prop = (
                    self.log_f_estimator(u, theta))
                return log_f_est

            if self.do_metropolis_update:
                thetas[s], log_f_est_curr, rejection = mcmc.metropolis_step(
                    thetas[s-1], log_f_est_curr, log_f_func_2, self.prng,
                    self.prop_sampler, self.prop_scales)
            else:
                thetas[s], log_f_est_curr, rejection = mcmc.met_hastings_step(
                    thetas[s-1], log_f_est_curr, log_f_func_2, self.prng,
                    self.prop_sampler, self.prop_scales, self.log_prop_density)
            if rejection:
                n_reject_2 += 1
            else:
                # if proposal accepted update current cached results
                cached_res_curr = self._cached_res_prop
        return thetas, (n_reject_1, n_reject_2)


class APMEllSSPlusMHSampler(BaseAdaptiveMHSampler):
    """ Auxiliary pseudo-marginal ESS + MH sampler.

    Sampler in the auxiliary pseudo-marginal MCMC framework which uses
    elliptical slice sampling updates for the random draws and Metropolis--
    Hastings updates for the parameter state.

    It is implicitly assumed the 'prior' :math:`q(u)` on the random draws is
    Gaussian in this case.
    """

    def __init__(self, log_f_estimator, log_prop_density, prop_sampler,
                 prop_scales, u_sampler, prng, max_slice_iters=1000):
        """ Auxiliary Pseudo-marginal ESS + MH sampler.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target distribution given current parameter state and
            random draws. Should have a call signature::
                log_f_est, cached_res_out =
                    log_f_estimator(u, theta, [, cached_res_in])
            where ``u`` is the vector of auxiliary random draws used in the
            density estimator, ``theta`` is the state vector (as ndarray) to
            estimate the density at, ``cached_res_in`` is an optional input
            which can be provided if cached intermediate results
            deterministically calculated from the ``theta`` which it is wished
            to estimate the density at have been stored from a previous call,
            potentially speeding subsequent estimates, ``log_f_est`` is the
            calculated log-density estimate and ``cached_res_out`` are
            intermediate cached results determinstically calculated from
            the specified ``theta`` which can be used in subsequent calls to
            potentially speed further estimates of the log density for this
            ``theta`` value (if ``cached_res_in`` was specified then
            ``cached_res_out == cached_res_in``).
        log_prop_density : function or callable object or None
            Function returning logarithm of parameter update proposal density
            at a given proposed parameter state given the current parameter
            state. Should have a call signature::
                log_prop_dens = log_prop_density(theta_prop, theta_curr)
            where ``theta_prop`` is proposed parameter state to evaluate the
            log proposal density at, ``theta_curr`` is the parameter state to
            condition the proposal density on and ``log_prop_dens`` is the
            returned log proposal density value. Alternatively ``None`` may
            be passed which indicates a symmetric proposal density in which
            case a Metropolis update will be made.
        prop_sampler : function or callable object
            Function which returns a proposed new parameter state drawn from
            proposal distribution given a current parameter state. Should have
            a call signature::
                theta_prop = prop_sampler(theta_curr, prop_scales)
            where ``theta_curr`` is the current parameter state vector (as a
            ndarray) which the proposal should be conditioned on,
            ``prop_scales`` is a ndarray of scale parameters for the proposal
            distribution (e.g. standard deviation for Gaussian proposals) and
            ``theta_prop`` is the returned random propsal distribution draw,
            again an ndarray.
        prop_scales : ndarray
            Array of values to initialise the scale parameters of the state
            proposal distribution to. If an initial adaptive run is performed
            by calling ``adaptive_run``, these parameters will be tuned to
            try to achieve an average accept rate in some prescribed interval.
        u_sampler : function or callable object
            Function which returns an independent sample from the 'prior'
            distribution on the random draws :math:`q(u)`.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        max_slice_iters : integer
            Maximum number of elliptical slice shrinking iterations to perform.
        """
        super(APMEllSSPlusMHSampler, self).__init__(prop_scales)
        self.log_f_estimator = log_f_estimator
        if log_prop_density is None:
            self.do_metropolis_update = True
        else:
            self.do_metropolis_update = False
            self.log_prop_density = log_prop_density
        self.prop_sampler = prop_sampler
        self.prop_scales = prop_scales
        self.prng = prng
        self.u_sampler = u_sampler
        self.max_slice_iters = max_slice_iters

    def elliptical_slice_sample_u_given_theta(self, u, log_f_est, log_f_func):
        """ Perform ESS on conditional density of random draws given state.

        Performs elliptical slice sampling conditional target density on
        auxiliary random draw variables given a parameter state.
        """
        v = self.u_sampler()
        return mcmc.elliptical_slice_step(u, log_f_est, log_f_func, self.prng,
                                          v, self.max_slice_iters)

    def get_samples(self, theta_init, n_sample, u_init=None):
        """ Perform a series of Markov chain updates.

        Parameters
        ----------
        theta_init : ndarray
            State to initialise parameters at, with shape ``(n_dim, )``.
        n_sample : integer
            Number of Markov chain updates to perform and so state samples to
            return.
        u_init : ndarray
            State to initialise random draws at. Optional, if not specified
            will be sampled from base density.

        Returns
        -------
        thetas : ndarray
            Two dimensional array of sampled chain states with shape
            ``(n_sample, n_dim)``.
        n_reject : integer or iterable
             The number of rejected proposed updates during the ``n_sample``
             updates.
        """
        if hasattr(theta_init, 'shape'):
            thetas = np.empty((n_sample, theta_init.shape[0]))
        else:
            thetas = np.empty(n_sample)
        thetas[0] = theta_init
        u = u_init if u_init is not None else self.u_sampler()
        log_f_est_curr, self._cached_res_curr = (
            self.log_f_estimator(u, theta_init))
        n_reject = 0
        for s in range(1, n_sample):
            ## Update u keeping theta fixed using ell-SS
            # As for this update only u will be changed, cached results
            # for current theta calculated in previous log_f_estimator call
            # can be reused, hence pass these values to estimator (with no
            # theta value being needed in this case) and use only first
            # return value (as second will be equal to cached_res_curr)
            log_f_func_1 = lambda v: (
                self.log_f_estimator(v, thetas[s-1], self._cached_res_curr)[0])
            u, log_f_est_curr = self.elliptical_slice_sample_u_given_theta(
                u, log_f_est_curr, log_f_func_1)
            ## Update theta keeping u fixed using MH

            def log_f_func_2(theta):
                # save cached results from estimator evaluation for proposed
                # theta update so this can be saved to be used in
                # final call of log_f_func in slice sampling routine will
                # always be accepted update so cached results will be correct
                log_f_est, self._cached_res_prop = (
                    self.log_f_estimator(u, theta))
                return log_f_est

            if self.do_metropolis_update:
                thetas[s], log_f_est_curr, rejection = mcmc.metropolis_step(
                    thetas[s-1], log_f_est_curr, log_f_func_2, self.prng,
                    self.prop_sampler, self.prop_scales)
            else:
                thetas[s], log_f_est_curr, rejection = mcmc.met_hastings_step(
                    thetas[s-1], log_f_est_curr, log_f_func_2, self.prng,
                    self.prop_sampler, self.prop_scales, self.log_prop_density)
            if rejection:
                n_reject += 1
            else:
                # if proposal accepted update current cached results
                self._cached_res_curr = self._cached_res_prop
        return thetas, n_reject


class BaseAPMMetIndPlusSliceSampler(object):
    """ Abstract auxiliary pseudo-marginal MI + SS sampler base class.

    Sampler in the auxiliary pseudo-marginal MCMC framework which uses
    Metropolis independence updates for the random draws and some form of
    linear slice sampling in updates for parameter state.
    """

    def __init__(self, log_f_estimator, u_sampler, prng, max_steps_out=0,
                 max_slice_iters=1000):
        """ Abstract auxiliary Pseudo-marginal MI + SS sampler base class.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target distribution given current parameter state and
            random draws. Should have a call signature::
                log_f_est, cached_res_out =
                    log_f_estimator(u, theta, [, cached_res_in])
            where ``u`` is the vector of auxiliary random draws used in the
            density estimator, ``theta`` is the state vector (as ndarray) to
            estimate the density at, ``cached_res_in`` is an optional input
            which can be provided if cached intermediate results
            deterministically calculated from the ``theta`` which it is wished
            to estimate the density at have been stored from a previous call,
            potentially speeding subsequent estimates, ``log_f_est`` is the
            calculated log-density estimate and ``cached_res_out`` are
            intermediate cached results determinstically calculated from
            the specified ``theta`` which can be used in subsequent calls to
            potentially speed further estimates of the log density for this
            ``theta`` value (if ``cached_res_in`` was specified then
            ``cached_res_out == cached_res_in``).
        u_sampler : function or callable object
            Function which returns an independent sample from the 'prior'
            distribution on the random draws :math:`q(u)`.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        max_steps_out : integer
            Maximum number of stepping out iterations to perform during slice
            sampling update (default 0).
        max_slice_iters : integer
            Maximum number of slice shrinking iterations to perform.
        """
        self.log_f_estimator = log_f_estimator
        self.u_sampler = u_sampler
        self.prng = prng
        self.max_steps_out = max_steps_out
        self.max_slice_iters = max_slice_iters

    def slice_step(self, x_curr, log_f_curr, log_f_func, w):
        """ Perform a linear slice sampling step.

        Simply wraps external module function passing in fixed object level
        arguments for more convenient calling.
        """
        return mcmc.linear_slice_step(x_curr, log_f_curr, log_f_func, w,
                                      self.prng, self.max_steps_out,
                                      self.max_slice_iters)

    def slice_sample_theta_given_u(self, theta, log_f_est, u):
        """ Perform SS on conditional density of state given random draws.

        Performs slice sampling along some line on conditional target density
        on parameter state given auxiliary random draw variables.

        Should be implemented by an derived class.
        """
        raise NotImplementedError()

    def get_samples(self, theta_init, n_sample, u_init=None):
        """ Perform a series of Markov chain updates.

        Parameters
        ----------
        theta_init : ndarray
            State to initialise parameters at, with shape ``(n_dim, )``.
        n_sample : integer
            Number of Markov chain updates to perform and so state samples to
            return.
        u_init : ndarray
            State to initialise random draws at. Optional, if not specified
            will be sampled from base density.

        Returns
        -------
        thetas : ndarray
            Two dimensional array of sampled chain states with shape
            ``(n_sample, n_dim)``.
        n_reject : integer or iterable
             The number of rejected proposed updates during the ``n_sample``
             updates.
        """
        if hasattr(theta_init, 'shape'):
            thetas = np.empty((n_sample, theta_init.shape[0]))
        else:
            thetas = np.empty((n_sample, 1))
        thetas[0] = theta_init
        u = u_init if u_init is not None else self.u_sampler()
        log_f_est_curr, self._cached_res_curr = (
            self.log_f_estimator(u, theta_init))
        n_reject = 0
        for s in range(1, n_sample):
            ## Update u keeping theta fixed using MI
            # As for this update only u will be changed, cached results
            # for current theta calculated in previous log_f_estimator call
            # can be reused, hence pass these values to estimator (with no
            # theta value being needed in this case) and use only first
            # return value (as second will be equal to cached_res_curr)
            log_f_func_1 = lambda v: (
                self.log_f_estimator(v, thetas[s-1], self._cached_res_curr)[0])
            u, log_f_est_curr, rejection = mcmc.metropolis_indepedence_step(
                u, log_f_est_curr, log_f_func_1, self.prng, self.u_sampler)
            if rejection:
                n_reject += 1
            ## Update theta given current u using SS
            # self.cached_res_curr also updated in this method
            thetas[s], log_f_est_curr = self.slice_sample_theta_gvn_u(
                thetas[s-1].copy(), log_f_est_curr, u)
        return thetas, n_reject


class BaseAPMEllSSPlusSliceSampler(object):
    """ Abstract auxiliary pseudo-marginal ESS + SS sampler base class.

    Sampler in the auxiliary pseudo-marginal MCMC framework which uses
    elliptical slice sampling updates for the random draws and some form of
    linear slice sampling in updates for parameter state.

    It is implicitly assumed the 'prior' :math:`q(u)` on the random draws is
    Gaussian in this case.
    """

    def __init__(self, log_f_estimator, u_sampler, prng, max_steps_out=0,
                 max_slice_iters=1000):
        """ Abstract auxiliary Pseudo-marginal ESS + SS sampler base class.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target distribution given current parameter state and
            random draws. Should have a call signature::
                log_f_est, cached_res_out =
                    log_f_estimator(u, theta, [, cached_res_in])
            where ``u`` is the vector of auxiliary random draws used in the
            density estimator, ``theta`` is the state vector (as ndarray) to
            estimate the density at, ``cached_res_in`` is an optional input
            which can be provided if cached intermediate results
            deterministically calculated from the ``theta`` which it is wished
            to estimate the density at have been stored from a previous call,
            potentially speeding subsequent estimates, ``log_f_est`` is the
            calculated log-density estimate and ``cached_res_out`` are
            intermediate cached results determinstically calculated from
            the specified ``theta`` which can be used in subsequent calls to
            potentially speed further estimates of the log density for this
            ``theta`` value (if ``cached_res_in`` was specified then
            ``cached_res_out == cached_res_in``).
        u_sampler : function or callable object
            Function which returns an independent sample from the 'prior'
            distribution on the random draws :math:`q(u)`.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        max_steps_out : integer
            Maximum number of stepping out iterations to perform during slice
            sampling update (default 0).
        max_slice_iters : integer
            Maximum number of slice shrinking iterations to perform (common
            to both elliptical and linear slice sampling updates).
        """
        self.log_f_estimator = log_f_estimator
        self.u_sampler = u_sampler
        self.prng = prng
        self.max_steps_out = max_steps_out
        self.max_slice_iters = max_slice_iters

    def slice_step(self, x_curr, log_f_curr, log_f_func, w):
        """ Perform a linear slice sampling step.

        Simply wraps external module function passing in fixed object level
        arguments for more convenient calling.
        """
        return mcmc.linear_slice_step(x_curr, log_f_curr, log_f_func, w,
                                      self.prng, self.max_steps_out,
                                      self.max_slice_iters)

    def elliptical_slice_sample_u_given_theta(self, u, log_f_est, log_f_func):
        """ Perform ESS on conditional density of random draws given state.

        Performs elliptical slice sampling conditional target density on
        auxiliary random draw variables given a parameter state.
        """
        v = self.u_sampler()
        return mcmc.elliptical_slice_step(u, log_f_est, log_f_func, self.prng,
                                          v, self.max_slice_iters)

    def slice_sample_theta_given_u(self, theta, log_f_est, u):
        """ Perform SS on conditional density of state given random draws.

        Performs slice sampling along some line on conditional target density
        on parameter state given auxiliary random draw variables.

        Should be implemented by an derived class.
        """
        raise NotImplementedError()

    def get_samples(self, theta_init, n_sample, u_init=None):
        """ Perform a series of Markov chain updates.

        Parameters
        ----------
        theta_init : ndarray
            State to initialise parameters at, with shape ``(n_dim, )``.
        n_sample : integer
            Number of Markov chain updates to perform and so state samples to
            return.
        u_init : ndarray
            State to initialise random draws at. Optional, if not specified
            will be sampled from base density.

        Returns
        -------
        thetas : ndarray
            Two dimensional array of sampled chain states with shape
            ``(n_sample, n_dim)``.
        """
        if hasattr(theta_init, 'shape'):
            thetas = np.empty((n_sample, theta_init.shape[0]))
        else:
            thetas = np.empty((n_sample, 1))
        thetas[0] = theta_init
        u = u_init if u_init is not None else self.u_sampler()
        log_f_est_curr, self._cached_res_curr = (
            self.log_f_estimator(u, theta_init))
        for s in range(1, n_sample):
            ## Update u keeping theta fixed using ell-SS
            log_f_func = lambda v: (
                self.log_f_estimator(v, thetas[s-1], self._cached_res_curr)[0]
                # second output will be equal to cached_res_curr as
                # not changing theta
            )
            u, log_f_est_curr = self.elliptical_slice_sample_u_given_theta(
                u, log_f_est_curr, log_f_func)
            ## Update theta given current u using SS
            # self.cached_res_curr also updated in this method
            thetas[s], log_f_est_curr = self.slice_sample_theta_gvn_u(
                thetas[s-1].copy(), log_f_est_curr, u)
        return thetas


class APMMetIndPlusSeqSliceSampler(BaseAPMMetIndPlusSliceSampler):
    """ Auxiliary pseudo-marginal MI + sequential-SS sampler.

    Sampler in the auxiliary pseudo-marginal MCMC framework which uses
    Metropolis independence updates for the random draws and sequential
    (over axes) slice sampling in updates for parameter state.
    """

    def __init__(self, log_f_estimator, u_sampler, prng, ws, max_steps_out=0,
                 max_slice_iters=1000):
        """ Auxiliary pseudo-marginal MI + sequential-SS sampler.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target distribution given current parameter state and
            random draws. Should have a call signature::
                log_f_est, cached_res_out =
                    log_f_estimator(u, theta, [, cached_res_in])
            where ``u`` is the vector of auxiliary random draws used in the
            density estimator, ``theta`` is the state vector (as ndarray) to
            estimate the density at, ``cached_res_in`` is an optional input
            which can be provided if cached intermediate results
            deterministically calculated from the ``theta`` which it is wished
            to estimate the density at have been stored from a previous call,
            potentially speeding subsequent estimates, ``log_f_est`` is the
            calculated log-density estimate and ``cached_res_out`` are
            intermediate cached results determinstically calculated from
            the specified ``theta`` which can be used in subsequent calls to
            potentially speed further estimates of the log density for this
            ``theta`` value (if ``cached_res_in`` was specified then
            ``cached_res_out == cached_res_in``).
        u_sampler : function or callable object
            Function which returns an independent sample from the 'prior'
            distribution on the random draws :math:`q(u)`.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        ws : ndarray
            Initial slice bracket widths to use when performing slice sampling
            sequentially on parameter state vector dimensions (i.e. `ws`
            should be same length as parameter state vector with a per
            dimension slice bracket width parameter being specified).
        max_steps_out : integer
            Maximum number of stepping out iterations to perform during slice
            sampling update (default 0).
        max_slice_iters : integer
            Maximum number of slice shrinking iterations to perform.
        """
        super(APMMetIndPlusSeqSliceSampler, self).__init__(
            log_f_estimator, u_sampler,  prng, max_steps_out, max_slice_iters)
        self.ws = ws

    def slice_sample_theta_gvn_u(self, theta, log_f_est, u):
        """ Perform seq-SS on conditional density of state given random draws.

        Performs slice sampling on conditional target density of each dimension
        of parameter state given rest of parameter state vector and auxiliary
        random draw variables, the dimension updates being peformed
        sequentially in a fixed ordinal ordering.
        """
        for j in range(len(theta)):
            x_curr = theta[j]

            def log_f_func(x):
                # keep saving cached results from new estimator evaluations
                # final call of log_f_func in slice sampling routine will
                # always be accepted update so cached results will be correct
                log_f_est_, self._cached_res_curr = (
                    self.log_f_estimator(u, np.r_[theta[:j], x, theta[j+1:]])
                )
                return log_f_est_

            x_new, log_f_est = self.slice_step(
                x_curr, log_f_est, log_f_func, self.ws[j])
            theta[j] = x_new
        return theta, log_f_est


class APMMetIndPlusRandDirSliceSampler(BaseAPMMetIndPlusSliceSampler):
    """ Auxiliary pseudo-marginal MI + random-direction-SS sampler.

    Sampler in the auxiliary pseudo-marginal MCMC framework which uses
    Metropolis independence updates for the random draws and slice sampling
    along a random direction in updates for parameter state.
    """

    def __init__(self, log_f_estimator, u_sampler, prng, slc_dir_and_w_sampler,
                 max_steps_out=0, max_slice_iters=1000):
        """ Auxiliary pseudo-marginal MI + random-direction-SS sampler.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target distribution given current parameter state and
            random draws. Should have a call signature::
                log_f_est, cached_res_out =
                    log_f_estimator(u, theta, [, cached_res_in])
            where ``u`` is the vector of auxiliary random draws used in the
            density estimator, ``theta`` is the state vector (as ndarray) to
            estimate the density at, ``cached_res_in`` is an optional input
            which can be provided if cached intermediate results
            deterministically calculated from the ``theta`` which it is wished
            to estimate the density at have been stored from a previous call,
            potentially speeding subsequent estimates, ``log_f_est`` is the
            calculated log-density estimate and ``cached_res_out`` are
            intermediate cached results determinstically calculated from
            the specified ``theta`` which can be used in subsequent calls to
            potentially speed further estimates of the log density for this
            ``theta`` value (if ``cached_res_in`` was specified then
            ``cached_res_out == cached_res_in``).
        u_sampler : function or callable object
            Function which returns an independent sample from the 'prior'
            distribution on the random draws :math:`q(u)`.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        slc_dir_and_w_sampler : function or callable object
            Function which returns a vector specifying a random direction in
            the parameter state space along to slice sample along with a
            corresponding initial slice bracket width for this direction.
            Should have a call signature::
                d, w = slc_dir_and_w_sampler()
            where ``d`` is a ndarray of same dimension as the parameter state
            and ``w`` is a (positive) floating point value specifying the
            corresponding initial slice bracket width parameter.
        max_steps_out : integer
            Maximum number of stepping out iterations to perform during slice
            sampling update (default 0).
        max_slice_iters : integer
            Maximum number of slice shrinking iterations to perform.
        """
        super(APMMetIndPlusRandDirSliceSampler, self).__init__(
            log_f_estimator, u_sampler, prng, max_steps_out, max_slice_iters)
        self.slc_dir_and_w_sampler = slc_dir_and_w_sampler

    def slice_sample_theta_gvn_u(self, theta, log_f_est, u):
        """ Perform rd-SS on conditional density of state given random draws.

        Performs slice sampling along a random direction on conditional target
        density of parameter state given auxiliary random draw variables.
        """
        d, w = self.slc_dir_and_w_sampler()

        def log_f_func(x):
            # keep saving cached results from new estimator evaluations
            # final call of log_f_func in slice sampling routine will
            # always be accepted update so cached results will be correct
            log_f_est_, self._cached_res_curr = (
                self.log_f_estimator(u, theta + x * d)
            )
            return log_f_est_

        x_new, log_f_est = self.slice_step(0., log_f_est, log_f_func, w)
        return theta + x_new * d, log_f_est


class APMEllSSPlusRandDirSliceSampler(BaseAPMEllSSPlusSliceSampler):
    """ Auxiliary pseudo-marginal ESS + random-direction-SS sampler.

    Sampler in the auxiliary pseudo-marginal MCMC framework which uses
    elliptical slice sampling updates for the random draws and slice sampling
    along a random direction in updates for parameter state.

    It is implicitly assumed the 'prior' :math:`q(u)` on the random draws is
    Gaussian in this case.
    """

    def __init__(self, log_f_estimator, u_sampler, prng, slc_dir_and_w_sampler,
                 max_steps_out=0, max_slice_iters=1000):
        """ Auxiliary pseudo-marginal ESS + random-direction-SS sampler.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target distribution given current parameter state and
            random draws. Should have a call signature::
                log_f_est, cached_res_out =
                    log_f_estimator(u, theta, [, cached_res_in])
            where ``u`` is the vector of auxiliary random draws used in the
            density estimator, ``theta`` is the state vector (as ndarray) to
            estimate the density at, ``cached_res_in`` is an optional input
            which can be provided if cached intermediate results
            deterministically calculated from the ``theta`` which it is wished
            to estimate the density at have been stored from a previous call,
            potentially speeding subsequent estimates, ``log_f_est`` is the
            calculated log-density estimate and ``cached_res_out`` are
            intermediate cached results determinstically calculated from
            the specified ``theta`` which can be used in subsequent calls to
            potentially speed further estimates of the log density for this
            ``theta`` value (if ``cached_res_in`` was specified then
            ``cached_res_out == cached_res_in``).
        u_sampler : function or callable object
            Function which returns an independent sample from the 'prior'
            distribution on the random draws :math:`q(u)`.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        slc_dir_and_w_sampler : function or callable object
            Function which returns a vector specifying a random direction in
            the parameter state space along to slice sample along with a
            corresponding initial slice bracket width for this direction.
            Should have a call signature::
                d, w = slc_dir_and_w_sampler()
            where ``d`` is a ndarray of same dimension as the parameter state
            and ``w`` is a (positive) floating point value specifying the
            corresponding initial slice bracket width parameter.
        max_steps_out : integer
            Maximum number of stepping out iterations to perform during slice
            sampling update (default 0).
        max_slice_iters : integer
            Maximum number of slice shrinking iterations to perform (common
            to both elliptical and linear slice sampling updates)
        """
        super(APMEllSSPlusRandDirSliceSampler, self).__init__(
            log_f_estimator, u_sampler, prng, max_steps_out, max_slice_iters)
        self.slc_dir_and_w_sampler = slc_dir_and_w_sampler

    def slice_sample_theta_gvn_u(self, theta, log_f_est, u):
        """ Perform rd-SS on conditional density of state given random draws.

        Performs slice sampling along a random direction on conditional target
        density of parameter state given auxiliary random draw variables.
        """
        d, w = self.slc_dir_and_w_sampler()

        def log_f_func(x):
            # keep saving cached results from new estimator evaluations
            # final call of log_f_func in slice sampling routine will
            # always be accepted update so cached results will be correct
            log_f_est_, self._cached_res_curr = (
                self.log_f_estimator(u, theta + x * d)
            )
            return log_f_est_

        x_new, log_f_est = self.slice_step(0., log_f_est, log_f_func, w)
        return theta + x_new * d, log_f_est


class APMEllSSPlusEllSSSampler(BaseAPMEllSSPlusSliceSampler):
    """ Auxiliary pseudo-marginal ESS + ESS sampler.

    Sampler in the auxiliary pseudo-marginal MCMC framework which uses
    elliptical slice sampling updates for both the random draws and parameter
    states.

    It is implicitly assumed the prior :math:`q(u)` on the random draws and
    the prior on the parameters :math:`p(\\theta)` are both Gaussian.
    """

    def __init__(self, log_f_estimator, u_sampler, theta_sampler, prng,
                 max_slice_iters=1000):
        """ Auxiliary pseudo-marginal ESS + ESS sampler.

        Parameters
        ----------
        log_f_estimator : function or callable object
            Function which returns an unbiased estimate of the log density
            of the target likelihood (i.e. without Gaussian prior on parameter
            state) given current parameter state and random draws. Should have
            a call signature::
                log_f_est, cached_res_out =
                    log_f_estimator(u, theta, [, cached_res_in])
            where ``u`` is the vector of auxiliary random draws used in the
            density estimator, ``theta`` is the state vector (as ndarray) to
            estimate the density at, ``cached_res_in`` is an optional input
            which can be provided if cached intermediate results
            deterministically calculated from the ``theta`` which it is wished
            to estimate the density at have been stored from a previous call,
            potentially speeding subsequent estimates, ``log_f_est`` is the
            calculated log-density estimate and ``cached_res_out`` are
            intermediate cached results determinstically calculated from
            the specified ``theta`` which can be used in subsequent calls to
            potentially speed further estimates of the log density for this
            ``theta`` value (if ``cached_res_in`` was specified then
            ``cached_res_out == cached_res_in``).
        u_sampler : function or callable object
            Function which returns an independent sample from the Gaussian
            prior distribution on the random draws :math:`q(u)`.
        theta_sampler : function or callable object
            Function which returns an independent sample from the Gaussian
            prior distribution on the parameters :math:`p(\\theta)`.
        prng : RandomState
            Pseudo-random number generator object (either an instance of a
            ``numpy`` ``RandomState`` or an object with an equivalent
            interface) used to randomly sample accept decisions in MH accept
            step.
        max_slice_iters : integer
            Maximum number of slice shrinking iterations to perform.
        """
        super(APMEllSSPlusEllSSSampler, self).__init__(
            log_f_estimator, u_sampler, prng, None, max_slice_iters)
        self.theta_sampler = theta_sampler

    def slice_sample_theta_gvn_u(self, theta, log_f_est, u):
        """ Perform ESS on conditional density of state given random draws.

        Performs elliptical slice sampling on conditional target
        density of parameter state given auxiliary random draw variables.
        """

        def log_f_func(theta):
            # keep saving cached results from new estimator evaluations
            # final call of log_f_func in slice sampling routine will
            # always be accepted update so cached results will be correct
            log_f_est_, self._cached_res_curr = (
                self.log_f_estimator(u, theta)
            )
            return log_f_est_

        v = self.theta_sampler()
        return mcmc.elliptical_slice_step(
            theta, log_f_est, log_f_func, self.prng, v, self.max_slice_iters)
