"""PyNEST Mesocircuit: Run Plotting
-----------------------------------

Create plots of the network activity of the mesocircuit.
"""

###############################################################################
# Import the necessary modules and setup the time measurements.
import os
import sys
import pickle
import h5py
import numpy as np
import core.plotting.plotting as plotting
import core.plotting.figures as figures
import core.helpers.parallelism_time as pt

##########################################################################
# Load simulation, network, stimulation, analysis and plotting parameters from
# files located in the folder provided as command line argument.

path_parameters = sys.argv[1]

dics = []
for dic in ['sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
    with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, ana_dict, plot_dict = dics

##########################################################################
# Instantiate a Plotting object.
# Load preprocessed data and pre-computed statistics.

pl = plotting.Plotting(sim_dict, net_dict, ana_dict, plot_dict)

d = {}
for datatype in np.append(ana_dict['datatypes_preprocess'],
                          ana_dict['datatypes_statistics']):
    all_datatype = 'all_' + datatype
    fn = os.path.join(sim_dict['path_processed_data'], all_datatype + '.h5')
    data = h5py.File(fn, 'r')
    d.update({all_datatype: data})

##########################################################################
# Plot figures and measure times.
# Time measurements are printed.

functions = [
    [figures.parameters,
     [pl]],

    [figures.raster,
     [pl, d['all_sptrains'], d['all_pos_sorting_arrays']]],

    [figures.statistics_overview,
     [pl, d['all_FRs'], d['all_LVs'], d['all_CCs_distances'], d['all_PSDs']]],

    [figures.corrcoef_distance,
     [pl, d['all_CCs_distances']]],

    [figures.spatial_snapshots,
     [pl, d['all_inst_rates_bintime_binspace']]],

    [figures.crosscorrelation_funcs_thalamic_pulses,
     [pl, d['all_CCfuncs_thalamic_pulses']]],

    [figures.instantaneous_firing_rates,
     [pl, d['all_sptrains_bintime']]],
]

pt.run_serial_functions_in_parallel(functions, os.path.basename(__file__))

# TODO close files
