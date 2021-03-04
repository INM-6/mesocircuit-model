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
import core.helpers.parallelism_time as pt

##########################################################################
# Get the number of threads per MPI process from the command line.
# Load simulation and network parameters.
local_num_threads = sys.argv[-1]

dics = []
for dic in ['sim_dict', 'net_dict']:
    with open(os.path.join('parameters', f'{dic}.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict = dics

###############################################################################
# Initialize the network with parameters, create and connect all nodes, and
# finally simulate.
# The times for a presimulation and the main simulation are taken
# independently. A presimulation is useful because the spike activity typically
# exhibits a startup transient. In benchmark simulations, this transient should
# be excluded from a time measurement of the state propagation phase. Besides,
# statistical measures of the spike activity should only be computed after the
# transient has passed.
# Time measurements are printed.

net = network.Network(sim_dict, net_dict, local_num_threads)

functions = [
    net.create,
    net.connect,
    [net.presimulate, [sim_dict['t_presim']]],
    [net.simulate, [sim_dict['t_sim']]],
]

pt.run_parallel_functions_sequentially(functions, os.path.basename(__file__))
