"""PyNEST Mesocircuit: Run Analysis
-----------------------------------

Analyze the spiking network activity of the mesocircuit.
"""

###############################################################################
# Import the necessary modules and setup the time measurements.
import os
import sys
import pickle
import core.analysis.spike_analysis as spike_analysis
import core.helpers.time_measurement as time_measurement

################################################################################
# Load simulation, network, stimulation and analysis parameters from files
# located in the folder provided as command line argument.

path_parameters = sys.argv[1]

dics = []
for dic in ['sim_dict', 'net_dict', 'stim_dict', 'ana_dict']:
    with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, stim_dict, ana_dict = dics

################################################################################
# Instantiate a SpikeAnalysis object, preprocess the data (spatial and temporal
# binning), and compute statistics.

sana = spike_analysis.SpikeAnalysis(sim_dict, net_dict, stim_dict, ana_dict)

logtime_data = [] # list for collecting time measurements

sana.preprocess_data(logtime=logtime_data)

sana.compute_statistics(logtime=logtime_data)

sana.merge_h5_files_populations(logtime=logtime_data)

################################################################################
# Summarize time measurements. 

time_measurement.print_times(os.path.basename(__file__), logtime_data)