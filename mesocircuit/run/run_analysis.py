"""PyNEST Mesocircuit: Run Analysis
-----------------------------------

Analyze the spiking network activity of the mesocircuit.
"""

###############################################################################
# Import the necessary modules.

import os
import sys
import mesocircuit.mesocircuit_framework as mesoframe
import mesocircuit.analysis.spike_analysis as spike_analysis
import mesocircuit.helpers.parallelism_time as pt

###############################################################################
# Instantiate a Mesocircuit object with parameters from the command line:
# the general data directory data_dir, the name of the experiment name_exp, and
# the ID of this parameterset ps_id.
# Previously evaluated parameters are loaded.

circuit = mesoframe.Mesocircuit(
    data_dir=sys.argv[-3], name_exp=sys.argv[-2], ps_id=sys.argv[-1],
    load_parameters=True)

##########################################################################
# Instantiate a SpikeAnalysis object, preprocess the data (spatial and temporal
# binning), and compute statistics.
# Time measurements are printed.

sana = spike_analysis.SpikeAnalysis(circuit)

functions = [
    sana.preprocess_data,
    sana.compute_statistics,
    sana.merge_h5_files_populations
]

pt.run_parallel_functions_sequentially(functions, os.path.basename(__file__))
