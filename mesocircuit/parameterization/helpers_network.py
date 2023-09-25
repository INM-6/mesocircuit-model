"""Helper Functions Network
---------------------------

Helper functions for deriving dependent network and stimulus parameters from
base parameter dictionaries.

"""

import numpy as np
import scipy
import copy


def derive_dependent_parameters(base_net_dict):
    """
    Derives network parameters which depend on the base parameters.
    Returns the dictionaries which serve as input to the Network class.

    Neuron and synapse numbers of thalamus are treated as a cortical population.

    Parameters
    ----------
    base_net_dict
        Dictionary with base network parameters.

    Returns
    -------
    net_dict
        Dictionary containing base and derived network parameters.
    """
    # network parameters
    net_dict = copy.copy(base_net_dict)

    # total number of populations (last one is TC)
    net_dict['num_pops'] = len(net_dict['populations'])

    # shape for connectivity matrices etc.
    pop_shape = (net_dict['num_pops'] - 1, net_dict['num_pops'])

    ############################################################################
    # set up matrices for spatial profiles and delays
    ############################################################################

    # decay parameters of exponential profile
    if np.all(net_dict['beta_exc_inh']):
        # matrix of decay parameters
        beta = get_exc_inh_matrix(
            net_dict['beta_exc_inh'][0],
            net_dict['beta_exc_inh'][1],
            net_dict['num_pops'])[:, :-1]  # thalamic value separate
    else:
        beta = net_dict['beta_unscaled'] * net_dict['beta_scaling']
    net_dict['beta'] = np.zeros(pop_shape)
    net_dict['beta'][:, :-1] = beta
    net_dict['beta'][:, -1] = net_dict['beta_th']

    # matrices for delays
    if net_dict['delay_type'] == 'normal':
        # matrix of mean delays
        net_dict['delay_matrix_mean'] = get_exc_inh_matrix(
            net_dict['delay_exc_mean'],
            net_dict['delay_inh_mean'],
            net_dict['num_pops'])

    elif net_dict['delay_type'] == 'linear':
        # matrix of delay offsets
        net_dict['delay_offset_matrix'] = get_exc_inh_matrix(
            net_dict['delay_offset_exc_inh'][0],
            net_dict['delay_offset_exc_inh'][1],
            net_dict['num_pops'])
        # matrix of propagation speeds
        net_dict['prop_speed_matrix'] = get_exc_inh_matrix(
            net_dict['prop_speed_exc_inh'][0],
            net_dict['prop_speed_exc_inh'][1],
            net_dict['num_pops'])
        # mean delays derived from linear parameters
        net_dict['delay_lin_eff_mean'], net_dict['delay_lin_eff_std'] = \
            get_delay_lin_effective(
                # extent of 1 mm2 for comparison with reference model
                1./np.sqrt(np.pi),
                net_dict['beta'],
                net_dict['delay_offset_matrix'],
                net_dict['prop_speed_matrix'])

    ############################################################################
    # PSCs from PSPs with default synaptic time constant
    ############################################################################

    # matrix of mean PSPs,
    # the mean PSP of the connection from L4E to L23E is doubled
    PSP_matrix_mean = get_exc_inh_matrix(
        net_dict['PSP_exc_mean'],
        net_dict['PSP_exc_mean'] * net_dict['g'],
        net_dict['num_pops'])
    PSP_matrix_mean[0, 2] = 2. * net_dict['PSP_exc_mean']

    # apply relative weight E to I
    PSP_matrix_mean[1::2, 0::2] *= net_dict['rel_weight_exc_to_inh']

    # conversion from PSPs to PSCs using the default synaptic time constant
    PSC_over_PSP_default = postsynaptic_potential_to_current(
        net_dict['neuron_params']['C_m'],
        net_dict['neuron_params']['tau_m'],
        net_dict['neuron_params']['tau_syn_default'])

    PSC_matrix_mean_tau_syn_default = \
        PSP_matrix_mean * PSC_over_PSP_default
    PSC_ext_tau_syn_default = \
        net_dict['PSP_exc_mean'] * PSC_over_PSP_default

    ############################################################################
    # neuron numbers and indegrees for full model
    ############################################################################

    # 1mm2 neuron number dependent on the base model
    num_neurons_1mm2 = np.zeros(net_dict['num_pops'])
    num_neurons_1mm2[:-1] = \
        net_dict['num_neurons_1mm2_' + net_dict['base_model']]
    num_neurons_1mm2[-1] = net_dict['num_neurons_th_1mm2']

    # 1mm2 external indegrees dependent on the base model.
    # thalamus does not have external indegrees
    ext_indegrees_1mm2 = net_dict['K_ext_' + net_dict['base_model']]

    # linear scaling of neuron numbers with square area
    area = net_dict['extent']**2
    full_num_neurons = num_neurons_1mm2 * area
    net_dict['full_num_neurons'] = np.round(full_num_neurons).astype(int)
    net_dict['full_num_neurons_sum'] = \
        np.round(np.sum(full_num_neurons)).astype(int)

    # in-degrees from connectivity matrix of 1mm2 model
    if net_dict['base_model'] == 'PD2014':
        conn_probs_1mm2 = np.zeros(pop_shape)
        conn_probs_1mm2[:, :-1] = net_dict['conn_probs_1mm2_PD2014']
        conn_probs_1mm2[:, -1] = net_dict['conn_probs_th_1mm2']

        # number of synapses of a full-scale 1mm2 network
        num_synapses_1mm2 = num_synapses_from_conn_probs(
            conn_probs_1mm2,
            num_neurons_1mm2,  # sources
            num_neurons_1mm2[:-1])  # targets, thalamus is only source

        indegrees_1mm2 = (num_synapses_1mm2 /
                          num_neurons_1mm2[:-1][:, np.newaxis])

    elif net_dict['base_model'] == 'SvA2018':
        num_synapses_th_1mm2 = num_synapses_from_conn_probs(
            net_dict['conn_probs_th_1mm2'][np.newaxis].T,  # column needed
            np.array([num_neurons_1mm2[-1]]),  # only thalamus
            num_neurons_1mm2[:-1])

        indegrees_th_1mm2 = (num_synapses_th_1mm2 /
                             num_neurons_1mm2[:-1][:, np.newaxis])

        indegrees_1mm2 = np.zeros(pop_shape)
        indegrees_1mm2[:, :-1] = net_dict['indegrees_1mm2_SvA2018']
        indegrees_1mm2[:, -1] = indegrees_th_1mm2.flatten()

    net_dict['indegrees_1mm2'] = np.round(indegrees_1mm2).astype(int)

    # mask radius (maximum distance of connections)
    mask_radius = net_dict['mask_scaling'] * net_dict['beta']
    # maximum mask radius is half of the extent
    mask_radius[mask_radius > net_dict['extent'] / 2.] = \
        net_dict['extent'] / 2.
    net_dict['mask_radius'] = mask_radius

    # in-degrees are scaled only if the extent is > 1 and
    # connect_method is 'fixedindegree_exp' or distr_indegree_exp;
    # otherwise the indegrees from the 1mm2 network are preserved
    if (net_dict['extent'] > 1. and net_dict['connect_method'] in [
            'fixedindegree_exp', 'distr_indegree_exp', 'distr_indegree_gauss']):
        # scale indegrees from disc of 1mm2 to disc of radius extent/2.
        if net_dict['K_area_scale_method'] == 'PD2014':
            net_dict['K_area_scaling'] = scale_indegrees_to_mask_radius(
                mask_radius=net_dict['mask_radius'],
                decay=np.ones_like(net_dict['beta']) * 0.3, profile='gaussian')
        elif net_dict['K_area_scale_method'] == 'beta':
            net_dict['K_area_scaling'] = scale_indegrees_to_mask_radius(
                mask_radius=net_dict['mask_radius'],
                decay=net_dict['beta'], profile='exponential')
        else:
            raise Exception('K_area_scale_method incorrect.')

        # elementwise multiplication because K_area_scaling is a matrix
        full_indegrees = np.multiply(
            indegrees_1mm2, net_dict['K_area_scaling'])
    else:
        full_indegrees = indegrees_1mm2

    # scale specific indegrees
    if len(net_dict['indegree_scaling']) > 0:
        for source, target, factor in net_dict['indegree_scaling']:
            full_indegrees[int(target)][int(source)] *= factor

    # adjust external indegrees to compensate for changed interal indegrees
    full_ext_indegrees = adjust_ext_indegrees_to_preserve_mean_input(
        indegrees_1mm2[:, :-1], full_indegrees[:, :-1],
        ext_indegrees_1mm2,
        net_dict['mean_rates_' + net_dict['base_model']],
        net_dict['bg_rate'],
        PSC_matrix_mean_tau_syn_default[:, :-1],
        PSC_ext_tau_syn_default)

    net_dict['full_indegrees'] = np.round(full_indegrees).astype(int)
    full_num_synapses = full_indegrees * full_num_neurons[:-1][:, np.newaxis]
    net_dict['full_num_synapses'] = np.round(full_num_synapses).astype(int)
    net_dict['full_num_synapses_sum'] = \
        np.round(np.sum(full_num_synapses)).astype(int)

    ############################################################################
    # DC input with compensation for potentially missing Poisson input
    ############################################################################

    if net_dict['poisson_input']:
        net_dict['full_DC_amp'] = np.zeros(net_dict['num_pops'] - 1)
    else:
        print('DC input compensates for missing Poisson input.')
        # uses default synaptic time constant
        net_dict['full_DC_amp'] = dc_input_compensating_poisson(
            net_dict['bg_rate'], full_ext_indegrees,
            net_dict['neuron_params']['tau_syn_default'],
            PSC_ext_tau_syn_default)

    ############################################################################
    # (down-)scale numbers of neurons and indegrees and finalize weights
    ############################################################################

    num_neurons = full_num_neurons * net_dict['N_scaling']
    indegrees = full_indegrees * net_dict['K_scaling']
    net_dict['num_neurons'] = np.round(num_neurons).astype(int)
    net_dict['indegrees'] = np.round(indegrees).astype(int)
    net_dict['num_synapses'] = np.round(full_num_synapses *
                                        net_dict['N_scaling'] *
                                        net_dict['K_scaling']).astype(int)
    ext_indegrees = full_ext_indegrees * net_dict['K_scaling']

    # adjust weights and DC amplitude if the indegree is scaled
    if net_dict['K_scaling'] != 1:
        # compute first with default synaptic time constants
        PSC_matrix_mean_tau_syn_default_scale, PSC_ext_tau_syn_default_scale, net_dict['DC_amp'] = \
            adjust_weights_and_input_to_indegree_scaling(
                full_indegrees,
                net_dict['K_scaling'],
                PSC_matrix_mean_tau_syn_default,
                PSC_ext_tau_syn_default,
                net_dict['neuron_params']['tau_syn_default'],
                net_dict['mean_rates_' + net_dict['base_model']],
                net_dict['full_DC_amp'],
                net_dict['poisson_input'],
                net_dict['bg_rate'],
                full_ext_indegrees)

    else:
        PSC_matrix_mean_tau_syn_default_scale = PSC_matrix_mean_tau_syn_default
        PSC_ext_tau_syn_default_scale = PSC_ext_tau_syn_default
        net_dict['DC_amp'] = net_dict['full_DC_amp']

    ############################################################################
    # scale external indegrees (changes are not compensated for)
    ############################################################################

    # population-specific changes
    if len(net_dict['ext_indegree_scaling']) > 0:
        for population, factor in net_dict['ext_indegree_scaling']:
            full_ext_indegrees[int(population)] *= factor
            ext_indegrees[int(population)] *= factor

    # global scaling factor
    full_ext_indegrees *= net_dict['ext_indegree_scaling_global']
    ext_indegrees *= net_dict['ext_indegree_scaling_global']

    net_dict['full_ext_indegrees'] = np.round(full_ext_indegrees).astype(int)
    net_dict['ext_indegrees'] = np.round(ext_indegrees).astype(int)

    ############################################################################
    # scale weights for changed excitatatory and inhibitory time constants
    ############################################################################

    net_dict['full_weight_matrix_mean'], net_dict['full_weight_ext'] = \
        adjust_weights_and_input_to_synaptic_time_constants(
            PSC_matrix_mean_tau_syn_default,
            PSC_ext_tau_syn_default,
            net_dict['neuron_params']['tau_syn_default'],
            net_dict['neuron_params']['tau_syn_ex'],
            net_dict['neuron_params']['tau_syn_in'],
            net_dict['num_pops'])

    net_dict['weight_matrix_mean'], net_dict['weight_ext'] = \
        adjust_weights_and_input_to_synaptic_time_constants(
            PSC_matrix_mean_tau_syn_default_scale,
            PSC_ext_tau_syn_default_scale,
            net_dict['neuron_params']['tau_syn_default'],
            net_dict['neuron_params']['tau_syn_ex'],
            net_dict['neuron_params']['tau_syn_in'],
            net_dict['num_pops'])

    ############################################################################
    # spatial connectivity
    ############################################################################

    # p0 is computed for non-fixed in-degrees
    # connectivity profile: p0 * exp(-r/beta)
    if net_dict['connect_method'] in [
            'distr_indegree_exp', 'distr_indegree_gauss']:
        if net_dict['connect_method'] == 'distr_indegree_exp':
            profile = 'exponential'
        elif net_dict['connect_method'] == 'distr_indegree_gauss':
            profile = 'gaussian'
        net_dict['p0_raw'], net_dict['p0'], net_dict['repeat_connect'] = \
            zero_distance_conn_prob_exp(num_neurons,
                                        indegrees,
                                        net_dict['extent'],
                                        net_dict['beta'],
                                        net_dict['mask_radius'],
                                        profile)
    else:
        net_dict['repeat_connect'] = np.ones_like(indegrees, dtype=int)

    # absolute radius for thalamic pulses
    if net_dict['thalamic_input_type'] == 'pulses':
        net_dict['th_radius'] = net_dict['th_rel_radius'] * net_dict['extent']

    return net_dict


def scale_indegrees_to_mask_radius(mask_radius, decay, profile):
    """
    Computes a matrix of factors to scale indegrees from a disc of area 1mm2 to
    a disc with radius of half of the extent.

    Parameters
    ----------
    mask_radius
        Matrix of mask radii. They are equal to the maximum distance of
        connections (in mm).
    decay
        Matrix of decay parameters (in mm).
    profile
        Function of spatial connectivity profile.
        Options are 'exponential' or 'gaussian'.

    Returns
    -------
    K_indegree_scaling
        Matrix of scaling factors to be applied to indegrees_1mm2.
    """

    def expression(radius, decay, profile):
        """
        int( r * profile(r,decay), r=0..R)
        Skip factor decay**2 in result because only fraction is used.
        """
        frac = radius / decay
        if profile == 'exponential':
            return 1. - np.exp(-frac) * (1. + frac)
        elif profile == 'gaussian':
            return 1. - np.exp(-0.5 * frac**2)
        else:
            raise Exception('Connectivity profile incorrect.')

    radius_1mm2 = 1. / np.sqrt(np.pi)

    K_indegree_scaling = (expression(mask_radius, decay, profile) /
                          expression(radius_1mm2, decay, profile))
    return K_indegree_scaling


def adjust_ext_indegrees_to_preserve_mean_input(
        indegrees_1mm2,
        full_indegrees,
        ext_indegrees_1mm2,
        mean_rates,
        bg_rate,
        PSC_matrix_mean,
        PSC_ext):
    """
    Computes external indegrees to adjusted to modified internal indegrees to
    preserve the mean input.

    Parameters
    ----------
    indegrees_1mm2
        Indegree matrix of the 1mm2 network without thalamus.
    full_indegrees
        Indegree matrix of the full network without thalamus.
    ext_indegrees_1mm2
        External indegrees of the 1mm2 network.
    mean_rates
        Mean firing rates of each population (in spikes/s).
    bg_rate
        Background firing rate (in spikes/s).
    PSC_matrix_mean
        Weight matrix (in pA) without thalamus.
    PSC_ext
        External weight (in pA).

    Returns
    -------
    full_ext_indegrees
        Adjusted external indegrees.
    """
    frac_psc = PSC_matrix_mean / PSC_ext
    frac_rates = mean_rates / bg_rate
    diff_indegrees = indegrees_1mm2 - full_indegrees

    diff_rec_inputs = np.multiply(frac_psc * frac_rates, diff_indegrees)
    sum_diff_rec_inputs = np.sum(diff_rec_inputs, axis=1)

    full_ext_indegrees = ext_indegrees_1mm2 + sum_diff_rec_inputs

    return full_ext_indegrees


def zero_distance_conn_prob_exp(
        num_neurons,
        indegrees,
        extent,
        decay,
        mask_radius,
        profile):
    """
    Computes the zero-distance connection probability and repeat factors for
    the connect_method 'distr_indegree_exp'.

    int (r * c_uni, r=0..R) = int (r * p0 * exp(-r/beta), r=0..R)
    with R = extent / 2

    Parameters
    ----------
    num_neurons
        Number of neurons.
    indegrees
        Indegree matrix.
    extent
        Side length (in mm) of square sheets where neurons are distributed.
    decay
        Matrix of decay parameters (in mm).
    mask_radius
        Matrix of mask radii (in mm).
    profile
        Function of spatial connectivity profile.
        Options are 'exponential' or 'gaussian'.

    Returns
    -------
    p0_raw
        Product of p0 and repeat_connect.
    p0
        Zero-distance connection probabilities used in one Connect() call.
    repeat_connect
        Factor for repeating the Connect() call.
    """
    # connection probability inside of mask with given radius
    # obtained by scaling neuron numbers with pi * R^2 / L^2
    num_potential_sources = num_neurons * np.pi * mask_radius**2 / extent**2
    conn_prob_uniform = indegrees / num_potential_sources

    frac = mask_radius / decay
    p0_raw = conn_prob_uniform * 0.5 * frac**2
    if profile == 'exponential':
        p0_raw *= (1. - np.exp(-frac) * (1. + frac))
    elif profile == 'gaussian':
        p0_raw *= (1. - np.exp(-0.5 * frac**2))
    else:
        raise Exception('Connectivity profile incorrect.')

    repeat_connect = np.ones_like(indegrees, dtype=int)
    p0 = copy.copy(p0_raw)

    # update the repeat factor and p0 if p0 raw exceeds 1
    for i in np.arange(len(p0_raw)):
        for j in np.arange(len(p0_raw[i])):
            if p0_raw[i][j] > 1:
                repeat_connect[i][j] = int(np.ceil(p0_raw[i][j]))
                p0[i][j] = p0_raw[i][j] / repeat_connect[i][j]

    return p0_raw, p0, repeat_connect


def get_exc_inh_matrix(val_exc, val_inh, num_pops):
    """
    Creates a matrix for excitatory and inhibitory values.

    The thalamus is assumed as an additional excitatory population,
    meaning that the network has num_pops source populations and num_pops-1
    target populations.

    Parameters
    ----------
    val_exc
        Excitatory value.
    val_inh
        Inhibitory value.
    num_pops
        Number of populations.

    Returns
    -------
    matrix
        A matrix of of size ((num_pops-1) x num_pops).

    """
    matrix = np.zeros((num_pops - 1, num_pops))
    matrix[:, 0:num_pops:2] = val_exc
    matrix[:, 1:num_pops:2] = val_inh
    return matrix


def get_delay_lin_effective(
        radius,
        beta,
        delay_offset_matrix,
        prop_speed_matrix):
    """
    Computes the effective mean and standard deviation of the linear delay.

    Uses Sheng (1985), Theorem 2.4 (a=b, c=0)

    Parameters
    ----------
    radius
        Radius (in mm) of disc for computing the effective delay.
    beta
        Matrix of decay parameters of exponential spatial profile (in mm).
    delay_offset_matrix
        Matrix of delay offsets (in ms).
    prop_speed_matrix
        Matrix of propagation speeds (in mm/ms).

    Returns
    -------
    delay_lin_eff_mean_matrix
        Matrix of effective mean delays (in ms).
    delay_lin_eff_std_matrix
        Matrix of effective standard deviation of delays (in ms).
    """

    # linear distance dependence of delay
    def delayfunc(r, d0, v):
        return d0 + r / v

    # exponential distance dependence of connectivity
    def connfunc(r, b):
        return np.exp(-r / b)

    def integrand(r, R, b, variable):
        alpha = 2. * np.arctan(np.sqrt((2. * R - r) / (2. * R + r)))
        return variable * connfunc(r, b) * r * (2. * alpha - np.sin(2. * alpha))

    def integrand_delay_mean(r, R, b, d0, v):
        return integrand(r, R, b, delayfunc(r, d0, v))

    def integrand_delay_var(r, R, b, d0, v, dmean):
        return integrand(r, R, b, (delayfunc(r, d0, v) - dmean)**2)

    def integrand_conn_norm(r, R, b):
        return integrand(r, R, b, 1)

    limits = [0., 2. * radius]  # integral bounds
    num_pops = len(delay_offset_matrix)

    delay_lin_eff_mean_matrix = np.zeros((num_pops, num_pops))
    delay_lin_eff_std_matrix = np.zeros((num_pops, num_pops))
    for i in np.arange(num_pops):
        for j in np.arange(num_pops):
            d0 = delay_offset_matrix[i, j]
            v = prop_speed_matrix[i, j]
            b = beta[i, j]

            I_delay_mean = scipy.integrate.quad(integrand_delay_mean,
                                                limits[0],
                                                limits[1],
                                                args=(radius, b, d0, v))[0]

            # normalization
            I_conn_norm = scipy.integrate.quad(integrand_conn_norm,
                                               limits[0],
                                               limits[1],
                                               args=(radius, b))[0]

            dmean = I_delay_mean / I_conn_norm

            I_delay_var = scipy.integrate.quad(integrand_delay_var,
                                               limits[0],
                                               limits[1],
                                               args=(radius, b, d0, v, dmean))[0]

            dvar = I_delay_var / I_conn_norm
            dstd = np.sqrt(dvar)

            delay_lin_eff_mean_matrix[i, j] = dmean
            delay_lin_eff_std_matrix[i, j] = dstd
    return delay_lin_eff_mean_matrix, delay_lin_eff_std_matrix


def num_synapses_from_conn_probs(
        conn_probs, num_neurons_source, num_neurons_target):
    """Computes the total number of synapses between two populations from
    connection probabilities.

    Paramters
    ---------
    conn_probs
        Matrix of connection probabilities.
    num_neurons_source
        Sizes of the source poulations.
    num_neurons_target
        Sizes of the target populations.

    Returns
    -------
    num_synapses
        Matrix of synapse numbers.
    """
    prod = np.outer(num_neurons_target, num_neurons_source)
    num_synapses = np.log(1. - conn_probs) / np.log((prod - 1.) / prod)

    return num_synapses


def postsynaptic_potential_to_current(C_m, tau_m, tau_syn):
    """Computes a factor to convert postsynaptic potentials to currents.

    The time course of the postsynaptic potential ``v`` is computed as
    :math: `v(t)=(i*h)(t)`
    with the exponential postsynaptic current
    :math:`i(t)=J\\mathrm{e}^{-t/\tau_\\mathrm{syn}}\\Theta (t)`,
    the voltage impulse response
    :math:`h(t)=\frac{1}{\tau_\\mathrm{m}}\\mathrm{e}^{-t/\tau_\\mathrm{m}}\\Theta (t)`,
    and
    :math:`\\Theta(t)=1` if :math:`t\\geq 0` and zero otherwise.

    The ``PSP`` is considered as the maximum of ``v``, i.e., it is
    computed by setting the derivative of ``v(t)`` to zero.
    The expression for the time point at which ``v`` reaches its maximum
    can be found in Eq. 5 of [1]_.

    The amplitude of the postsynaptic current ``J`` corresponds to the
    synaptic weight ``PSC``.

    References
    ----------
    .. [1] Hanuschkin A, Kunkel S, Helias M, Morrison A and Diesmann M (2010)
           A general and efficient method for incorporating precise spike times
           in globally time-driven simulations.
           Front. Neuroinform. 4:113.
           DOI: `10.3389/fninf.2010.00113 <https://doi.org/10.3389/fninf.2010.00113>`__.

    Parameters
    ----------
    C_m
        Membrane capacitance (in pF).
    tau_m
        Membrane time constant (in ms).
    tau_syn
        Synaptic time constant (in ms).

    Returns
    -------
    PSC_over_PSP
        Conversion factor to be multiplied to a `PSP` (in mV) to obtain a `PSC`
        (in pA).

    """
    sub = 1. / (tau_syn - tau_m)
    pre = tau_m * tau_syn / C_m * sub
    frac = (tau_m / tau_syn) ** sub

    PSC_over_PSP = 1. / (pre * (frac**tau_m - frac**tau_syn))
    return PSC_over_PSP


def dc_input_compensating_poisson(
        bg_rate, K_ext, tau_syn_default, PSC_ext):
    """ Computes DC input if no Poisson input is provided to the mesocircuit.

    Parameters
    ----------
    bg_rate
        Rate of external Poisson generators (in spikes/s).
    K_ext
        External indegrees.
    tau_syn_default
        Default synaptic time constant (in ms).
    PSC_ext
        External weight (in pA).

    Returns
    -------
    DC
        DC input (in pA) which compensates lacking Poisson input.
    """
    DC = bg_rate * K_ext * PSC_ext * tau_syn_default * 0.001
    return DC


def adjust_weights_and_input_to_indegree_scaling(
        full_indegrees,
        K_scaling,
        PSC_matrix_mean,
        PSC_ext,
        tau_syn_default,
        full_mean_rates,
        DC_amp,
        poisson_input,
        bg_rate,
        K_ext):
    """ Adjusts weights and external input to scaling of indegrees.

    The recurrent and external weights are adjusted to the scaling
    of the indegrees. Extra DC input is added to compensate for the
    scaling in order to preserve the mean and variance of the input.

    Parameters
    ----------
    full_indegrees
        Indegree matrix of the full-scale network.
    K_scaling
        Scaling factor for indegrees.
    PSC_matrix_mean
        Weight matrix (in pA).
    PSC_ext
        External weight (in pA).
    tau_syn_default
        Default synaptic time constant (in ms).
    full_mean_rates
        Firing rates of the full network (in spikes/s).
    DC_amp
        DC input current (in pA).
    poisson_input
        True if Poisson input is used.
    bg_rate
        Firing rate of Poisson generators (in spikes/s).
    K_ext
        External indegrees.

    Returns
    -------
    PSC_matrix_new
        Adjusted weight matrix (in pA).
    PSC_ext_new
        Adjusted external weight (in pA).
    DC_amp_new
        Adjusted DC input (in pA).

    """
    PSC_matrix_new = PSC_matrix_mean / np.sqrt(K_scaling)
    PSC_ext_new = PSC_ext / np.sqrt(K_scaling)

    # recurrent input of full network without thalamus
    input_rec = np.sum(PSC_matrix_mean[:, :-1]
                       * full_indegrees[:, :-1]
                       * full_mean_rates, axis=1)

    DC_amp_new = DC_amp \
        + 0.001 * tau_syn_default * (1. - np.sqrt(K_scaling)) * input_rec

    if poisson_input:
        input_ext = PSC_ext * K_ext * bg_rate
        DC_amp_new += \
            tau_syn_default * 0.001 * (1. - np.sqrt(K_scaling)) * input_ext

    return PSC_matrix_new, PSC_ext_new, DC_amp_new


def adjust_weights_and_input_to_synaptic_time_constants(
        PSC_matrix_mean,
        PSC_ext,
        tau_syn_default,
        tau_syn_ex,
        tau_syn_in,
        num_pops):
    """
    Preserves mean input upon changed synapti time constants by weight changes.

    Note that the variance of the input is not preserved.

    Parameters
    ----------
    PSC_matrix_mean
        Weight matrix (in pA).
    PSC_ext
        External weight (in pA).
    tau_syn_default
        Default synaptic time constant (in ms).
    tau_syn_ex
        Excitatory synaptic time constant (in ms).
    tau_syn_in
        Inhibitory synaptic time constant (in ms).
    num_pops
        Number of populations.

    Returns
    -------
    PSC_matrix_new
        Adjusted weight matrix (in pA).
    PSC_ext_new
        Adjusted external weight (in pA).
    """
    # fractions of default synaptic time constants divided by excitatory and
    # inhibitory ones
    frac_tau_syn = get_exc_inh_matrix(tau_syn_default / tau_syn_ex,
                                      tau_syn_default / tau_syn_in,
                                      num_pops)

    # PSCs taking into account changes in synaptic time constants
    PSC_matrix_new = PSC_matrix_mean * frac_tau_syn
    PSC_ext_new = PSC_ext * tau_syn_default / tau_syn_ex
    return PSC_matrix_new, PSC_ext_new
