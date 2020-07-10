import os
import parameters as ps

local_sim_dict = {'computer': 'local',
                  'print_time': True}

jureca_sim_dict = {} # TODO

# main dictionary as used by evaluate_parameterspaces() in helpers.py
ps_dicts = {

    'local_downscale': {
        'sim_dict': {**local_sim_dict,
                     **{'data_path': os.path.join(os.getcwd(),
                                                  'data_local_downscale')}},
        'net_dict': {
            'N_scaling': 0.1,
            'K_scaling': 0.1,
            'delay_type': 'normal',
            'extent': 1.},
    }
}