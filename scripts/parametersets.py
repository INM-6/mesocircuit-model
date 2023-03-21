import numpy as np

local_sim_dict = {
    'print_time': True}

### model-specific definitions ###

net_dict_mesocircuit_MAMV1 = {
    'base_model': 'SvA2018',
    'g': -11.,
    'neuron_params': {'tau_syn_ex': 2., 'tau_syn_in': 8.},
    'indegree_scaling': np.array([[5, 4, 0.75], [3, 3, 1.8], [7, 7, 0.9]]),
    'ext_indegree_scaling': np.array([[0, 1.15], [2, 1.05]]),
    'ext_indegree_scaling_global': 1.04,
}

net_dict_microcircuit_MAMV1 = {
    'base_model': 'SvA2018',
    'g': -11.,
    'delay_type': 'normal',
    'connect_method': 'fixedtotalnumber',
    'extent': 1.}

net_dict_microcircuit_PD = {
    'base_model': 'PD2014',
    'delay_type': 'normal',
    'connect_method': 'fixedtotalnumber',
    'extent': 1.}

##########################################################################

ps_dicts = {

    # main mesocircuit based on MAM V1 of Schmidt & van Albada (2018)
    'mesocircuit_MAMV1': {
        'net_dict': {
            **net_dict_mesocircuit_MAMV1,
        },
        'sys_dict': {'hpc': {'network': {'num_nodes': 16}}}
    },

    # main mesocircuit with spatially confined stimulus
    'mesocircuit_MAMV1_evoked': {
        'net_dict': {
            **net_dict_mesocircuit_MAMV1,
            'thalamic_input': True,
        },
        'sys_dict': {'hpc': {'network': {'num_nodes': 16}}}
    },

    # Potjans & Diesmann (2014) microcircuit
    'microcircuit_PD': {
        'net_dict': {
            **net_dict_microcircuit_PD,
        },
        'sys_dict': {'hpc': {'network': {'num_nodes': 1}}}
    },

    # local mesocircuit
    'local_mesocircuit_MAMV1': {
        'sim_dict': {
            **local_sim_dict,
        },
        'net_dict': {
            **net_dict_mesocircuit_MAMV1,
            'N_scaling': 0.005,
            'K_scaling': 0.5,
        },
    },

    # MAM V1 microcircuit
    'microcircuit_MAMV1': {
        'net_dict': {
            **net_dict_microcircuit_MAMV1,
        },
        'sys_dict': {'hpc': {'network': {'num_nodes': 1}}}
    },

    # local Potjans & Diesmann (2014) microcircuit
    'local_microcircuit_PD': {
        'sim_dict': {
            **local_sim_dict,
        },
        'net_dict': {
            **net_dict_microcircuit_PD,
            'N_scaling': 0.1,
            'K_scaling': 0.1,
        },
    },

    # mesocircuit based on Potjans & Diesmann (2014)
    'mesocircuit_PD': {},  # base parameters

    # local mesocircuit based on Potjans & Diesmann (2014)
    'local_mesocircuit_PD': {
        'net_dict': {
            'N_scaling': 0.008,
            'K_scaling': 0.1,
        },
    },
}
