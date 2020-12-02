import os
import parameters as ps

# definitions for local simulations
local_data_path = os.path.join(os.getcwd(), 'data')

local_sim_dict = {
    'computer': 'local',
    'print_time': True,
    'num_mpi_per_node': 2,
    'local_num_threads': 4}

local_ana_dict = {
    'computer': 'local',
    # 'num_mpi_per_node': 1,
    }

local_plot_dict = {
    'computer': 'local',
    # 'num_mpi_per_node': 1,
    }

local_sim_ana_plot_dicts = {
    'sim_dict': local_sim_dict,
    'ana_dict': local_ana_dict,
    'plot_dict': local_plot_dict,
    }


# model-specific definitions
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

################################################################################

# main dictionary as used by evaluate_parameterspaces() in helpers.py
ps_dicts = {
    'microcircuit': {
        'sim_dict': {
            'num_nodes': 1,
            },
        'net_dict': {
            **net_dict_microcircuit,
            },
        },

    'local_microcircuit': {
        ** local_sim_ana_plot_dicts,
        'net_dict': {
            **net_dict_microcircuit,
            'N_scaling': 0.1,
            'K_scaling': 0.1,
               },
        },

    'macaqueV1': {
        'sim_dict': {
            'num_nodes': 1,
            },
        'net_dict': {
            **net_dict_macaqueV1,
            },
        },

    'local_macaqueV1': {
        **local_sim_ana_plot_dicts,
        'net_dict': {
            **net_dict_macaqueV1,
            'N_scaling': 0.1,
            'K_scaling': 0.1,
            },
        },

    'local_mesocircuit': {
        **local_sim_ana_plot_dicts,
        'net_dict': {
            'N_scaling': 0.008,
            'K_scaling': 0.1,
            },
        },
    }
