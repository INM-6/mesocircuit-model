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
import core.helpers.parallelism_time as pt

##########################################################################
# Load simulation, network, stimulation and analysis parameters.

dics = []
for dic in ['sim_dict', 'net_dict', 'ana_dict']:
    with open(os.path.join('parameters', f'{dic}.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, ana_dict = dics

##########################################################################
# Instantiate a SpikeAnalysis object, preprocess the data (spatial and temporal
# binning), and compute statistics.
# Time measurements are printed.

sana = spike_analysis.SpikeAnalysis(sim_dict, net_dict, ana_dict)

functions = [
    sana.preprocess_data,
    sana.compute_statistics,
    sana.merge_h5_files_populations
]

pt.run_parallel_functions_sequentially(functions, os.path.basename(__file__))
