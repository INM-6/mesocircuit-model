"""PyNEST Mesocircuit: Simulation Parameters
--------------------------------------------

A dictionary with parameters defining the simulation.

"""

import os

sim_dict = {
    # The full simulation time is the sum of a presimulation time and the main
    # simulation time.
    # presimulation time (in ms)
    't_presim': 500.0,
    # simulation time (in ms)
    't_sim': 1000.0,
    # resolution of the simulation (in ms)
    'sim_resolution': 0.1,
    # list of recording devices, default is 'spike_detector'. A 'voltmeter' can
    # be added to record membrane voltages of the neurons. Nothing will be
    # recorded if an empty list is given.
    'rec_dev': ['spike_detector'],
    # path to save the output data
    'data_path': os.path.join(os.getcwd(), 'data/'),
    # masterseed for NEST and NumPy
    'master_seed': 55,
    # number of threads per MPI process
    'local_num_threads': 1,
    # recording interval of the membrane potential (in ms)
    'rec_V_int': 1.0,
    # if True, data will be overwritten,
    # if False, a NESTError is raised if the files already exist
    'overwrite_files': True,
    # print the time progress. This should only be used when the simulation
    # is run on a local machine.
    'print_time': True}
