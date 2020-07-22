"""PyNEST Mesocircuit: Run Analysis
-------------------------------------

Analyze the spiking network activity of the mesocircuit.
"""

###############################################################################
# Import the necessary modules and start the time measurements.
import os
import sys
import pickle
from mpi4py import MPI
import analysis_spike
import time
time_start = time.time()

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

################################################################################
# Load simulation, network, stimulation and analysis parameters from files
# located in the folder provided as command line argument.

path_parameters = sys.argv[1]

dics = []
for dic in ['sim_dict', 'net_dict', 'stim_dict', 'ana_dict']:
    with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, stim_dict, ana_dict = dics

sana = analysis_spike.SpikeAnalysis(sim_dict, net_dict, stim_dict, ana_dict)
time_init = time.time()




time_stop = time.time()

print(
    '\nTimes of Rank {}:\n'.format(RANK) +
    '  Total analysis time:  {:.3f} s\n'.format(
        time_stop - time_start) +
    '  Time init: {:.3f} s\n'.format(
        time_init - time_start)
    )