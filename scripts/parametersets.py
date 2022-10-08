import numpy as np

local_sim_dict = {
    'print_time': True}

### model-specific definitions ###
net_dict_microcircuit = {
    'base_model': 'PD2014',
    'delay_type': 'normal',
    'connect_method': 'fixedtotalnumber',
    'extent': 1.}

net_dict_macaqueV1 = {
    'base_model': 'SvA2018',
    'g': -11.,
    'delay_type': 'normal',
    'connect_method': 'fixedtotalnumber',
    'extent': 1.}

net_dict_mesomacaqueV1 = {
    'base_model': 'SvA2018',
    'g': -11.
}

net_dict_favorite = {
    'base_model': 'SvA2018',
    'g': -11.,
    'neuron_params': {'tau_syn_ex': 2., 'tau_syn_in': 8.},
    'indegree_scaling': np.array([[5, 4, 0.75], [3, 3, 1.8], [7, 7, 0.9]]),
    'ext_indegree_scaling': np.array([[0, 1.15], [2, 1.1]]),
}

##########################################################################

# main dictionary as used by evaluate_parameterspaces() in helpers.py
ps_dicts = {
    'base': {},

    'favorite': {
        'net_dict': {
            **net_dict_favorite,
        },
        'sys_dict': {'hpc': {'network': {'num_nodes': 16}}}
    },

    'favorite_evoked': {
        'net_dict': {
            **net_dict_favorite,
            'thalamic_input': 'pulses',
        },
        'sys_dict': {'hpc': {'network': {'num_nodes': 16}}}
    },

    'microcircuit': {
        'net_dict': {
            **net_dict_microcircuit,
        },
        'sys_dict': {'hpc': {'network': {'num_nodes': 1}}}
    },

    'local_microcircuit': {
        'net_dict': {
            **net_dict_microcircuit,
            'N_scaling': 0.1,
            'K_scaling': 0.1,
        },
    },

    'macaqueV1': {
        'net_dict': {
            **net_dict_macaqueV1,
        },
    },

    'local_macaqueV1': {
        'net_dict': {
            **net_dict_macaqueV1,
            'N_scaling': 0.1,
            'K_scaling': 0.1,
        },
    },

    'mesomacaqueV1': {
        'net_dict': {
            **net_dict_mesomacaqueV1,
        },
    },

    'local_mesomacaqueV1': {
        'sim_dict': {
            **local_sim_dict,
        },
        'net_dict': {
            **net_dict_mesomacaqueV1,
            'N_scaling': 0.008,
            'K_scaling': 0.1,
        },
    },

    'local_mesocircuit': {
        'sim_dict': {
            **local_sim_dict,
        },
        'net_dict': {
            'N_scaling': 0.008,
            'K_scaling': 0.1,
        },
    },
}
