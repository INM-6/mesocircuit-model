"""PyNEST Mesocircuit: Run Simulation
-------------------------------------

Run a simulation of the mesocircuit model.
"""

###############################################################################
# Import the necessary modules and start the time measurements.

import os
import sys
import pickle
import nest
import network
import time
time_start = time.time()

################################################################################
# Load simulation, network and stimulation parameters from files located in the
# folder provided as command line argument.

path_parameters = sys.argv[1]

dics = []
for dic in ['sim_dict', 'net_dict', 'stim_dict']:
    with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, stim_dict = dics

###############################################################################
# Initialize the network with simulation, network and stimulation parameters,
# then create and connect all nodes, and finally simulate.
# The times for a presimulation and the main simulation are taken
# independently. A presimulation is useful because the spike activity typically
# exhibits a startup transient. In benchmark simulations, this transient should
# be excluded from a time measurement of the state propagation phase. Besides,
# statistical measures of the spike activity should only be computed after the
# transient has passed.

net = network.Network(sim_dict, net_dict, stim_dict)
time_network = time.time()

net.create()
time_create = time.time()

net.connect()
time_connect = time.time()

net.simulate(sim_dict['t_presim'])
time_presimulate = time.time()

net.simulate(sim_dict['t_sim'])
time_simulate = time.time()

###############################################################################
# Summarize time measurements. Rank 0 usually takes longest because of the
# data evaluation and print calls.

print(
    '\nTimes of Rank {}:\n'.format(
        nest.Rank()) +
    '  Total network time:  {:.3f} s\n'.format(
        time_simulate -
        time_start) +
    '  Time to initialize:  {:.3f} s\n'.format(
        time_network -
        time_start) +
    '  Time to create:      {:.3f} s\n'.format(
        time_create -
        time_network) +
    '  Time to connect:     {:.3f} s\n'.format(
        time_connect -
        time_create) +
    '  Time to presimulate: {:.3f} s\n'.format(
        time_presimulate -
        time_connect) +
    '  Time to simulate:    {:.3f} s\n'.format(
        time_simulate -
        time_presimulate))
