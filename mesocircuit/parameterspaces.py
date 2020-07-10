import os
import parameters as ps

ps_dicts = {

    'local_downscale': {
        'sim_dict': {
            'computer': 'local',
            'print_time': True,
            'data_path': os.path.join(os.getcwd(), 'data_local_downscale')},
        'net_dict': {
            'N_scaling': 0.01,
            'K_scaling': 0.1},
    }
}



# TODO sim_dict for JURECA 