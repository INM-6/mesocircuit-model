"""Run simulation
-----------------

Run a simulation of the mesocircuit model.
"""

###############################################################################
# Import the necessary modules.

import os
import sys
import mesocircuit.mesocircuit_framework as mesoframe
import mesocircuit.simulation.network as network
import mesocircuit.helpers.parallelism_time as pt

###############################################################################
# Instantiate a Mesocircuit object with parameters from the command line:
# the general data directory data_dir, the name of the experiment name_exp, and
# the ID of this parameterset ps_id.
# Previously evaluated parameters are loaded.

circuit = mesoframe.Mesocircuit(
    data_dir=sys.argv[-3], name_exp=sys.argv[-2], ps_id=sys.argv[-1],
    load_parameters=True)

###############################################################################
# Get the number of threads per MPI process from the command line.
local_num_threads = sys.argv[-4]

###############################################################################
# Initialize the network with the Mesocircuit and specify functions for
# creating and connecting all nodes, and finally simulating.
# The times for a presimulation and the main simulation are taken
# independently.
# Time measurements are printed.

net = network.Network(circuit, local_num_threads)

functions = [
    net.create,
    net.connect,
    [net.presimulate, [circuit.sim_dict['t_presim']]],
    [net.simulate, [circuit.sim_dict['t_sim']]],
]

###############################################################################
# The defined functions are inherently parallel and we execute each one of them
# after the other.

pt.run_parallel_functions_sequentially(functions, os.path.basename(__file__))
