"""PyNEST Mesocircuit: Run Plotting
-----------------------------------

Create plots of the network activity of the mesocircuit.
"""

###############################################################################
# Import the necessary modules and start the time measurements.
import os
import sys
import pickle
from mpi4py import MPI
import plotting
import time
time_start = time.time()

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

################################################################################
# Load simulation, network, stimulation, analysis and plotting parameters from
# files located in the folder provided as command line argument.

path_parameters = sys.argv[1]

dics = []
for dic in ['sim_dict', 'net_dict', 'stim_dict', 'ana_dict', 'plot_dict']:
    with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, stim_dict, ana_dict, plot_dict = dics

################################################################################
# Instantiate a Plotting object.

pl = plotting.Plotting(sim_dict, net_dict, stim_dict, ana_dict, plot_dict)
time_init = time.time()

if RANK == 0:
    pl.fig_raster()




time_stop = time.time()

################################################################################
# Print times.

print(
    '\nTimes of Rank {}:\n'.format(RANK) +
    '  Total plotting time:  {:.3f} s\n'.format(
        time_stop - time_start) #+
    #'  Time init: {:.3f} s\n'.format(
    #    time_init - time_start) +
    #'  Time preprocess: {:.3f} s\n'.format(
    #    time_preprocess - time_init) +
    #'  Time statistics: {:.3f} s\n'.format(
    #    time_stop - time_preprocess)
    )