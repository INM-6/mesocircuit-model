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
    #'num_mpi_per_node': 1,
    }

local_plot_dict = {
    'computer': 'local',
    #'num_mpi_per_node': 1,
    }

# model-specific definitions
net_dict_microcircuit = {
    'delay_type': 'normal',
    'connect_method': 'fixedtotalnumber',
    'extent': 1.}

################################################################################

# main dictionary as used by evaluate_parameterspaces() in helpers.py
ps_dicts = {
    'microcircuit': {
        'sim_dict': {
            'num_nodes': 2,
        },
        'net_dict': {
            **net_dict_microcircuit,
        },
    },

    'local_microcircuit': {
        'sim_dict': {
            **local_sim_dict,
            **{'data_path': local_data_path + '/local_microcircuit'}},
        'net_dict': {
            **net_dict_microcircuit,
            **{'N_scaling': 0.1,
               'K_scaling': 0.1}},
        'ana_dict': {
            **local_ana_dict},
        'plot_dict': {
            **local_plot_dict},
        },

    'local_mesocircuit': {
        'sim_dict': {
            **local_sim_dict,
            **{'data_path': local_data_path + '/local_mesocircuit'}},
        'net_dict': {
            'N_scaling': 0.008,
            'K_scaling': 0.1},
            #'beta_scaling': ps.ParameterRange([3., 5.])},
        'ana_dict': {
            **local_ana_dict},
        'plot_dict': {
            **local_plot_dict},
    }
}
