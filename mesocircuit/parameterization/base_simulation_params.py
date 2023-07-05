"""Simulation parameters
------------------------

A dictionary with parameters defining the simulation.

"""

import os

sim_dict = {
    # print the time progress. This should only be used when the simulation
    # is run on a local machine.
    'print_time': False,
    # file name for node ids
    'fname_nodeids': 'population_nodeids.dat',
    # if True, data will be overwritten,
    # if False, a NESTError is raised if the files already exist
    'overwrite_files': True,
    # random number generator seed
    'rng_seed': 55,

    # The full simulation time is the sum of a presimulation time and the main
    # simulation time.
    # presimulation time (in ms)
    # a good choice for consistency is to choose it equal to
    # ana_dict['t_transient'] for ignoring the spikes during analysis
    't_presim': 1000.0,
    # simulation time (in ms)
    't_sim': 1000.0,
    # resolution of the simulation (in ms)
    'sim_resolution': 0.1,

    # list of recording devices, default is 'spike_recorder'. A 'voltmeter' can
    # be added to record membrane voltages of the neurons. Nothing will be
    # recorded if an empty list is given.
    'rec_dev': ['spike_recorder'],
    # recording interval of the membrane potential (in ms)
    'rec_V_int': 1.0,
}
