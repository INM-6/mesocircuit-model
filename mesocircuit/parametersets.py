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
    'g': -11.,
}

##########################################################################

# main dictionary as used by evaluate_parameterspaces() in helpers.py
ps_dicts = {
    'base': {},

    'microcircuit': {
        'net_dict': {
            **net_dict_microcircuit,
        },
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
