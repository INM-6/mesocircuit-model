"""PyNEST Mesocircuit: Run Simulation
-------------------------------------

Run a simulation of the mesocircuit model.
"""

###############################################################################
# Import the necessary modules and setup the time measurements.

import os
import sys
import pickle
import nest
import core.simulation.network as network
import core.helpers.time_measurement as time_measurement

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

logtime_data = [] # list for collecting time measurements

net.create(logtime=logtime_data)

net.connect(logtime=logtime_data)

net.presimulate(sim_dict['t_presim'], logtime=logtime_data)

net.simulate(sim_dict['t_sim'], logtime=logtime_data)

###############################################################################
# Summarize time measurements. Rank 0 usually takes longest because of the
# data evaluation and print calls.

time_measurement.print_times(os.path.basename(__file__), logtime_data)
