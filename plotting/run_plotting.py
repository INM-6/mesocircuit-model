"""PyNEST Mesocircuit: Run Plotting
-----------------------------------

Create plots of the network activity of the mesocircuit.
"""

###############################################################################
# Import the necessary modules and start the time measurements.
import os
import sys
import pickle
import numpy as np
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
# Load preprocessed data and pre-computed statistics.

pl = plotting.Plotting(sim_dict, net_dict, stim_dict, ana_dict, plot_dict)

for datatype in np.append(ana_dict['datatypes_preprocess'],
                          ana_dict['datatypes_statistics']):
    all_datatype = 'all_' + datatype
    fn = os.path.join(sim_dict['path_processed_data'], all_datatype + '.h5')
    data = h5py.File(fn, 'r')
    globals().update({all_datatype: data}) # TODO improve?
time_init = time.time()

################################################################################
# Plot figures.

if RANK == 0:
    pl.fig_raster(all_sptrains, all_pos_sorting_arrays)
if RANK == 1:
    pl.fig_statistics_overview(all_sptrains)



# TODO close files
time_stop = time.time()

################################################################################
# Print times.

print(
    '\nTimes of Rank {}:\n'.format(RANK) +
    '  Total plotting time:  {:.3f} s\n'.format(
        time_stop - time_start) +
    '  Time init: {:.3f} s\n'.format(
        time_init - time_start) +
    '  Time plotting: {:.3f} s\n'.format(
        time_stop - time_init)
    )
