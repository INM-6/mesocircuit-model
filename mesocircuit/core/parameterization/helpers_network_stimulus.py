"""PyNEST Mesocircuit: Helper Functions Network Stimulus
--------------------------------------------------------

Helper functions for deriving dependent network and stimulus parameters from
base parameter dictionaries.

"""

import numpy as np
import copy

def derive_dependent_parameters(base_net_dict, base_stim_dict):
    """
    Derives network and stimulus parameters which depend on the base parameters.
    Returns the dictionaries which serve as input to the Network class.

    Parameters
    ----------
    base_net_dict
        Dictionary with base network parameters.
    base_stim_dict
        Dictionary with base stimulus parameters.

    Returns
    -------
    net_dict
        Dictionary containing base and derived network parameters.
    stim_dict
        Dictionary containing base and derived stimulus parameters.
    """
    # network parameters
    net_dict = copy.copy(base_net_dict)

    # number of cortical populations
    net_dict['num_pops'] = len(net_dict['populations'])

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

    # decay parameters of exponential profile
    if net_dict['beta_exc_inh']:
        # matrix of decay parameters
        net_dict['beta'] = get_exc_inh_matrix(
            net_dict['beta_exc_inh'][0],
            net_dict['beta_exc_inh'][1],
            net_dict['num_pops'])
    else:
        net_dict['beta'] = net_dict['beta_unscaled'] * net_dict['beta_scaling']

    # matrix of mean PSPs
    # the mean PSP of the connection from L4E to L23E is doubled
    PSP_matrix_mean = get_exc_inh_matrix(
        net_dict['PSP_exc_mean'],
        net_dict['PSP_exc_mean'] * net_dict['g'],
        net_dict['num_pops'])
    PSP_matrix_mean[0, 2] = 2. * net_dict['PSP_exc_mean']

    # conversion from PSPs to PSCs
    PSC_over_PSP = postsynaptic_potential_to_current(
        net_dict['neuron_params']['C_m'],
        net_dict['neuron_params']['tau_m'],
        net_dict['neuron_params']['tau_syn'])
    PSC_matrix_mean = PSP_matrix_mean * PSC_over_PSP
    PSC_ext = net_dict['PSP_exc_mean'] * PSC_over_PSP

    # 1mm2 neuron number dependent on the base model
    num_neurons_1mm2 = net_dict['num_neurons_1mm2_' + net_dict['base_model']]
    # 1mm2 external indegrees dependent on the base model
    K_ext = net_dict['K_ext_' + net_dict['base_model']]

    # linear scaling of neuron numbers with square area
    area = net_dict['extent']**2
    full_num_neurons = num_neurons_1mm2 * area
    net_dict['full_num_neurons'] = np.round(full_num_neurons).astype(int)
    net_dict['full_num_neurons_sum'] = \
        np.round(np.sum(full_num_neurons)).astype(int)

    # TODO adjust when parameters are final
    if net_dict['base_model'] == 'PD2014':
        # number of synapses of a full-scale 1mm2 network
        num_synapses_1mm2 = num_synapses_from_conn_probs(
            net_dict['conn_probs_1mm2_PD2014'],
            num_neurons_1mm2,
            num_neurons_1mm2)

        # average indegrees in 1mm2 network
        indegrees_1mm2 = (num_synapses_1mm2 /
                          num_neurons_1mm2[:,np.newaxis])

    elif net_dict['base_model'] == 'SvA2018':
        indegrees_1mm2 = net_dict['indegrees_1mm2_SvA2018']

    net_dict['indegrees_1mm2'] = np.round(indegrees_1mm2).astype(int)
                      
    # indegrees are scaled only if connect_method is 'fixedindegree_exp';
    # otherwise the indegrees from the 1mm2 network are preserved
    if net_dict['connect_method'] in ['fixedindegree_exp', 'distr_indegree_exp']:
        # scale indegrees from disc of 1mm2 to disc of radius extent/2.
        net_dict['K_area_scaling'] = scale_indegrees_to_extent(
            net_dict['extent'], net_dict['beta'])

        # elementwise multiplication because K_area_scaling is a matrix
        full_indegrees = np.multiply(indegrees_1mm2, net_dict['K_area_scaling'])
    else:
        full_indegrees = indegrees_1mm2
    net_dict['full_indegrees'] = np.round(full_indegrees).astype(int)
    full_num_synapses = full_indegrees * full_num_neurons[:, np.newaxis]
    net_dict['full_num_synapses'] = np.round(full_num_synapses).astype(int)
    net_dict['full_num_synapses_sum'] = \
        np.round(np.sum(full_num_synapses)).astype(int)

    # (down-)scale numbers of neurons and synapses
    num_neurons = full_num_neurons * net_dict['N_scaling']
    indegrees = full_indegrees * net_dict['K_scaling']
    net_dict['num_neurons'] = np.round(num_neurons).astype(int)
    net_dict['indegrees'] = np.round(indegrees).astype(int)
    net_dict['num_synapses'] = np.round(full_num_synapses *
                                        net_dict['N_scaling'] *
                                        net_dict['K_scaling']).astype(int)
    net_dict['ext_indegrees'] = np.round(K_ext *
                                         net_dict['K_scaling']).astype(int)

    # DC input compensates for potentially missing Poisson input
    if net_dict['poisson_input']:
        DC_amp = np.zeros(net_dict['num_pops'])
    else:
        if nest.Rank() == 0:
            print('DC input compensates for missing Poisson input.\n')
        DC_amp = dc_input_compensating_poisson(
            net_dict['bg_rate'], K_ext,
            net_dict['neuron_params']['tau_syn'],
            PSC_ext)

    # adjust weights and DC amplitude if the indegree is scaled
    if net_dict['K_scaling'] != 1:
        PSC_matrix_mean, PSC_ext, DC_amp = \
            adjust_weights_and_input_to_synapse_scaling(
                full_indegrees,
                net_dict['K_scaling'],
                PSC_matrix_mean, PSC_ext,
                net_dict['neuron_params']['tau_syn'],
                net_dict['mean_rates_' + net_dict['base_model']],
                DC_amp,
                net_dict['poisson_input'],
                net_dict['bg_rate'], K_ext)

    # p0 is computed for non-fixed in-degrees
    # connectivity profile: p0 * exp(-r/beta)
    if net_dict['connect_method'] == 'distr_indegree_exp':
        net_dict['p0_raw'], net_dict['p0'], net_dict['repeat_connect'] = \
            zero_distance_conn_prob_exp(num_neurons,
                                        indegrees,
                                        net_dict['extent'],
                                        net_dict['beta'])
    else:
        net_dict['repeat_connect'] = np.ones_like(indegrees,dtype=int)


    # store final parameters in dictionary
    net_dict['weight_matrix_mean'] = PSC_matrix_mean
    net_dict['weight_ext'] = PSC_ext
    net_dict['DC_amp'] = DC_amp


    # stimulus parameters
    stim_dict = copy.copy(base_stim_dict)

    # thalamic input
    if stim_dict['thalamic_input']:
        # linear scaling of thalamic neuron numbers with area
        # (note that thalamic neurons are not downscaled with N_scaling)
        num_th_neurons = stim_dict['num_th_neurons_1mm2'] * area
        stim_dict['num_th_neurons'] = \
            np.round(num_th_neurons).astype(int)

        # TODO synapses not finished
        # number of thalamic synapses
        num_th_synapses_1mm2 = num_synapses_from_conn_probs(
            stim_dict['conn_probs_th_1mm2'],
            stim_dict['num_th_neurons_1mm2'],
            num_neurons_1mm2)[0]

        # TODO just for first test
        num_th_synapses = num_th_synapses_1mm2


        stim_dict['weight_th'] = stim_dict['PSP_th'] * PSC_over_PSP
        if net_dict['K_scaling'] != 1:
            num_th_synapses *= net_dict['K_scaling']
            stim_dict['weight_th'] /= np.sqrt(net_dict['K_scaling'])
        stim_dict['num_th_synapses'] = np.round(num_th_synapses).astype(int)

    return net_dict, stim_dict


def scale_indegrees_to_extent(extent, beta):
    """
    Computes a matrix of factors to scale indegrees from a disc of area 1mm2 to
    a disc with radius of half of the extent. The latter corresponds to the
    radius of the cut-off mask used by the net_dict['connect_method'] is
    'fixedindegree_exp'.

    Parameters
    ----------
    extent
        Side length (in mm) of square sheets where neurons are distributed.
    beta
        Matrix of decay parameters of exponential spatial profile (in mm).

    Returns
    -------
    K_indegree_scaling
        Matrix of scaling factors to be applied to indegrees_1mm2.
    """

    def expression(beta, radius):
        frac = radius / beta
        return 1. - np.exp(-frac) * (1. + frac)

    radius_1mm2 = 1. / np.sqrt(np.pi)
    radius_area = extent / 2.

    K_indegree_scaling = (expression(beta, radius_area) /
                          expression(beta, radius_1mm2)) 

    return K_indegree_scaling


def zero_distance_conn_prob_exp(num_neurons, indegrees, extent, beta):
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
    beta
        Matrix of decay parameters of exponential spatial profile (in mm).

    Returns
    -------
    p0_raw
        Product of p0 and repeat_connect.
    p0
        Zero-distance connection probabilities used in one Connect() call.
    repeat_connect
        Factor for repeating the Connect() call.
    """
    # connection probability inside of mask with radius extent / 2.
    # without spatial profile.
    # pi * (extent/2)**2 / extent**2 = pi /4
    conn_prob_uniform = indegrees / (num_neurons * np.pi/4.)

    radius = extent / 2.
    p0_raw = (conn_prob_uniform * 0.5 * radius**2 /
             (beta * (beta - np.exp(-radius / beta) * (beta + radius))))

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
    """ Creates a matrix for excitatory and inhibitory values.

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
        A matrix of of size (num_pops x num_pops).

    """
    matrix = np.zeros((num_pops, num_pops))
    matrix[:, 0:num_pops:2] = val_exc
    matrix[:, 1:num_pops:2] = val_inh
    return matrix


def num_synapses_from_conn_probs(conn_probs, popsize1, popsize2):
    """Computes the total number of synapses between two populations from
    connection probabilities.

    Here it is irrelevant which population is source and which target.

    Paramters
    ---------
    conn_probs
        Matrix of connection probabilities.
    popsize1
        Size of first poulation.
    popsize2
        Size of second population.

    Returns
    -------
    num_synapses
        Matrix of synapse numbers.
    """
    prod = np.outer(popsize1, popsize2)
    num_synapses = np.log(1. - conn_probs) / np.log((prod - 1.) / prod)
    return num_synapses


def postsynaptic_potential_to_current(C_m, tau_m, tau_syn):
    """ Computes a factor to convert postsynaptic potentials to currents.

    The time course of the postsynaptic potential ``v`` is computed as
    :math: `v(t)=(i*h)(t)`
    with the exponential postsynaptic current
    :math:`i(t)=J\mathrm{e}^{-t/\tau_\mathrm{syn}}\Theta (t)`,
    the voltage impulse response
    :math:`h(t)=\frac{1}{\tau_\mathrm{m}}\mathrm{e}^{-t/\tau_\mathrm{m}}\Theta (t)`,
    and
    :math:`\Theta(t)=1` if :math:`t\geq 0` and zero otherwise.

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


def dc_input_compensating_poisson(bg_rate, K_ext, tau_syn, PSC_ext):
    """ Computes DC input if no Poisson input is provided to the mesocircuit.

    Parameters
    ----------
    bg_rate
        Rate of external Poisson generators (in spikes/s).
    K_ext
        External indegrees.
    tau_syn
        Synaptic time constant (in ms).
    PSC_ext
        Weight of external connections (in pA).

    Returns
    -------
    DC
        DC input (in pA) which compensates lacking Poisson input.
    """
    DC = bg_rate * K_ext * PSC_ext * tau_syn * 0.001
    return DC


def adjust_weights_and_input_to_synapse_scaling(
        full_indegrees,
        K_scaling,
        mean_PSC_matrix,
        PSC_ext,
        tau_syn,
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
    mean_PSC_matrix
        Weight matrix (in pA).
    PSC_ext
        External weight (in pA).
    tau_syn
        Synaptic time constant (in ms).
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
    PSC_matrix_new = mean_PSC_matrix / np.sqrt(K_scaling)
    PSC_ext_new = PSC_ext / np.sqrt(K_scaling)

    # recurrent input of full network
    input_rec = np.sum(mean_PSC_matrix * full_indegrees * full_mean_rates,
                       axis=1)

    DC_amp_new = DC_amp \
        + 0.001 * tau_syn * (1. - np.sqrt(K_scaling)) * input_rec

    if poisson_input:
        input_ext = PSC_ext * K_ext * bg_rate
        DC_amp_new += 0.001 * tau_syn * (1. - np.sqrt(K_scaling)) * input_ext
    return PSC_matrix_new, PSC_ext_new, DC_amp_new