"""PyNEST Mesocircuit: Network Parameters
-----------------------------------------

A dictionary with base network and neuron parameters is enhanced with derived
parameters.

"""

import numpy as np

net_dict = {
    # factor to scale the number of neurons
    'N_scaling': 1.,
    # factor to scale the indegrees
    'K_scaling': 1.,
    # neuron model
    'neuron_model': 'iaf_psc_exp',
    # names of the simulated neuronal populations
    'populations': np.array(
        ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I', 'TC']),
    # base model used for num_neurons_1mm2, conn_probs_1mm2 / indegrees_1mm2,
    # mean_rates, and K_ext
    # options are
    # 'PD2014': Potjans & Diesmann (2014)
    # 'SvA2018': Schmidt ... van Albada (2018)
    'base_model': 'PD2014',
    # number of neurons in the different populations (same order as
    # 'populations') of network covering 1mm2
    'num_neurons_1mm2_PD2014':
        np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948]),
    'num_neurons_1mm2_SvA2018':
        np.array([47386, 13366, 70387, 17597, 20740, 4554, 19839, 4063]),
    # number of thalamic neurons
    'num_neurons_th_1mm2': 902,
    # connection probabilities (the first index corresponds to the targets
    # and the second to the sources) of network covering 1mm2
    'conn_probs_1mm2_PD2014':
        np.array(
            [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0., 0.0076, 0.],
             [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0., 0.0042, 0.],
             [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.],
             [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0., 0.1057, 0.],
             [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.],
             [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.],
             [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443]]),
    'indegrees_1mm2_SvA2018':
        np.array(
            [[1062.05780665956,
              516.693556417054,
              481.639619708102,
              227.662598516076,
              77.2223337087846,
              2.58278612545931E-05,
              56.9825572758422,
              0.002547719312792],
             [1441.22871932965,
              413.650339113414,
              348.511756043073,
              142.765724065597,
              183.906753011762,
              2.8078192398913E-05,
              28.4797152892552,
              0.001724818102904],
             [76.101553634937,
              17.0532541730215,
              548.050632633596,
              382.50205031852,
              16.7235740718305,
              0.154558136653199,
              324.103963496397,
              0.002796270620683],
             [718.027297810924,
              8.54769496114012,
              876.372124238087,
              457.88686095073,
              7.1585610485389,
              2.55118224065622E-05,
              780.639330341388,
              0.000999471255078],
             [1042.30016686051,
              181.530391691357,
              557.991338204657,
              15.9348532849194,
              204.095347075258,
              227.182779372723,
              143.127021832045,
              0.006843860953766],
             [568.214412020203,
              78.1506995350578,
              282.524967334319,
              5.33709891502392,
              146.307555909299,
              187.593064274647,
              64.0388656508819,
              0.001459301922844],
             [159.401948962628,
              19.9785159873622,
              227.441084484738,
              46.0107686142342,
              138.832688971888,
              10.5525583810149,
              287.535665222409,
              355.683175924637],
             [368.658922577975,
              2.77868306292702,
              31.990783610677,
              2.64327372576867,
              67.4774463423829,
              4.20258522768281,
              478.851911854799,
              220.365998044097]]),
    # use the total number of synapses from the arxiv preprint
    'use_old_full_num_synapses': False,
    # connection probabilities from thalamus
    'conn_probs_th_1mm2':
        np.array([0.0, 0.0, 0.0983, 0.0619, 0.0, 0.0, 0.0512, 0.0196]),
    # scaling factors for indegrees; the working point is preserved by adjusting
    # the external indegrees.
    # entries have the format: [source, target, factor]
    'indegree_scaling': np.array([]),
    # mean rates of the different populations in the non-scaled version of the
    # mesocircuit (in spikes/s; same order as in 'populations');
    # necessary for the scaling of the network.
    # The values were optained by running this PyNEST microcircuit with 12 MPI
    # processes and both 'N_scaling' and 'K_scaling' set to 1.
    # TODO update with full-scale mesocircuit, or distinguish between full mean
    # rates or mean rates of full 1mm2
    'mean_rates_PD2014':
        np.array([0.943, 3.026, 4.368, 5.882, 7.733, 8.664, 1.096, 7.851]),
    # TODO recorded with 1mm2 macaqueV1, but probably not final
    'mean_rates_SvA2018':
        np.array([0.15709281, 1.6874907, 2.4200633, 2.8021822,
                  3.6009161, 4.2362757, 2.209184, 4.16761]),
    # mean amplitude of excitatory postsynaptic potential (in mV)
    'PSP_exc_mean': 0.15,
    # relative standard deviation of the weight
    'weight_rel_std': 0.1,
    # relative inhibitory weight
    'g': -4.,
    # relative weight E->I
    'rel_weight_exc_to_inh': 1.,

    # delay type, options are:
    # 'normal': normally distributed delays as in the original microcircuit
    # 'linear': constant offset, linearly distance-dependent and with normally
    #           distributed variability(default)
    'delay_type': 'linear',

    # delay parameters for the 'delay_type' set to 'normal'
    # mean delay of excitatory connections (in ms)
    'delay_exc_mean': 1.5,
    # mean delay of inhibitory connections (in ms)
    'delay_inh_mean': 0.75,
    # relative standard deviation of the delay of excitatory and
    # inhibitory connections
    'delay_rel_std': 0.5,

    # delay parameters for the 'delay_type' set to 'linear'
    # constant part of the linear delay (in ms)
    'delay_offset_exc_inh': np.array([0.5, 0.5]),
    # propagation speed (in mm/ms)
    'prop_speed_exc_inh': np.array([0.3, 0.3]),

    # turn Poisson input on or off (True or False)
    # if False: DC input is applied for compensation
    'poisson_input': True,
    # indegree of external connections to the different populations (same order
    # as in 'populations')
    'K_ext_PD2014': np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]),
    'K_ext_SvA2018': 10. / 8. * np.array([1267, 1251, 1255, 1246, 1430, 1250, 1777, 1254]),
    # overwrites external indegrees with the ones from the arxiv preprint
    'use_old_external_indegrees': False,
    # rate of the Poisson generator (in spikes/s)
    # scaling factor for external indegrees; changes are not compensated for.
    # entries have the format: [population, factor]
    'ext_indegree_scaling': np.array([]),
    # global scaling factor for external indegrees (applied to all populations)
    'ext_indegree_scaling_global': 1.,
    'bg_rate': 8.,
    # delay from the Poisson generator to the network (in ms)
    'delay_poisson': 1.5,

    # initial conditions for the membrane potential, options are:
    # 'original': uniform mean and standard deviation for all populations as
    #             used in earlier implementations of the model
    # 'optimized': population-specific mean and standard deviation, allowing a
    #              reduction of the initial activity burst in the network
    #              (default)
    # TODO parameters for optimized are currently from microcircuit
    'V0_type': 'optimized',
    # parameters of the neuron model
    'neuron_params': {
        # membrane potential average for the neurons (in mV)
        'V0_mean': {'original': -58.0,
                    'optimized': np.array([-68.28, -63.16, -63.33, -63.45,
                                           -63.11, -61.66, -66.72, -61.43])},
        # standard deviation of the average membrane potential (in mV)
        'V0_std': {'original': 10.0,
                   'optimized': np.array([5.36, 4.57, 4.74, 4.94,
                                          4.94, 4.55, 5.46, 4.48])},
        # reset membrane potential of the neurons (in mV)
        'E_L': -65.0,
        # threshold potential of the neurons (in mV)
        'V_th': -50.0,
        # membrane potential after a spike (in mV)
        'V_reset': -65.0,
        # membrane capacitance (in pF)
        'C_m': 250.0,
        # membrane time constant (in ms)
        'tau_m': 10.0,
        'tau_syn_default': 0.5,
        # time constants of postsynaptic currents (in ms)
        'tau_syn_ex': 0.5,
        'tau_syn_in': 0.5,
        # refractory period of the neurons after a spike (in ms)
        't_ref': 2.0},

    # method for scaling the in-degrees if extent > 1
    # 'PD2014': Gaussian decay with sigma 0.3 mm
    # 'beta': exponential decay with the same beta as used for connections
    # 'old': from old mesocircuit by scaling connection probabilities,
    #        includes conn_prob_modifications
    'K_area_scale_method': 'beta',
    # method to draw recurrent network connections, options are:
    # 'fixedtotalnumber': same routine as in the original microcircuit, i.e.,
    #                     fixed total number of connections, no space
    # 'fixedindegree': fixed indegree, no space
    # 'fixedindegree_exp': fixed indegree, distance-dependent connection
    #                      probabilities with an exponential profile, uses the
    #                      decay parameter 'beta'
    # 'distr_indegree_exp': distributed indegrees, distance-dependent connection
    #                      probabilities with an exponential profile, uses the
    #                      decay parameter 'beta' (default)
    # 'distr_indegree_gauss': TODO only temporary, uses beta as sigma
    'connect_method': 'distr_indegree_exp',
    # decay parameter of exponential profile (in mm),
    # used if 'connect_medhod' is 'fixedindegree_exp',
    # default values extracted from Reimann2017, Supplement 1, Figure S2,
    # format:   E->E   I->E
    #           E->I   I->I
    'beta_unscaled': np.tile([[0.232, 0.161],
                              [0.125, 0.120]], (4, 4)),
    # scaling factor applied to all elements of beta_unscaled.
    # The final beta is beta_unscaled * beta_scaling.
    'beta_scaling': 1.,

    # scaling factor for mask to be multiplied with the respective beta
    'mask_scaling': 5.,

    # TODO maybe remove
    # If beta_exh_inh is not False, it must be a list with excitatory and
    # inhibitory decay parameters [beta_exc, beta_inh] which will be used to
    # override the matrix beta above.
    'beta_exc_inh': False,

    # beta from thalamic population
    'beta_th': 0.1,

    # side length (in mm) of square sheets in which neurons are randomly
    # distributed
    'extent': 4.,

    # parameters for external stimulus
    # whether to use thalamic input
    # if False: thalamic neurons are created and connected, but they are not
    #           active
    'thalamic_input': False,

    # type of thalamic input (applied if thalamic_input = True), options are:
    # 'poisson':    persistent thalamic poisson input for a given duration to
    #               all thalamic neurons
    # 'pulses':     repetitive pulses from stimulating thalamic neurons  in the
    #               center of the network
    'thalamic_input_type': 'pulses',

    # thalamic_input_type = 'poisson'
    # start of the thalamic input (in ms)
    'th_start': 1000.0,
    # duration of the thalamic input (in ms)
    'th_duration': 10.0,
    # rate of the thalamic input (in spikes/s)
    'th_rate': 120.0,

    # thalamic_input_type = 'pulses'
    # only thalamic neurons within a circle in the center are stimulated.
    # the radius is th_rel_radius * extent
    'th_rel_radius': 0.1,
    # time of first pulse (in ms)
    'th_pulse_start': 1000.0,
    # pulse interval (in ms)
    'th_interval': 100.0,
    # delay between the pulse spike generator and the thalamic neurons
    'th_delay_pulse_generator': 1.0,

    # optional DC input
    # turn DC input on or off (True or False)
    'dc_input': False,
    # start of the DC input (in ms)
    'dc_start': 650.0,
    # duration of the DC input (in ms)
    'dc_dur': 100.0,
    # amplitude of the DC input (in pA); final amplitude is population-specific
    # and will be obtained by multiplication with 'K_ext'
    'dc_amp': 0.3
}
