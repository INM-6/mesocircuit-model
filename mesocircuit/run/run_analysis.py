"""Run analysis
---------------

Analyze the spiking network activity of the mesocircuit.
"""

###############################################################################
# Import the necessary modules.

import os
import sys
import mesocircuit.mesocircuit_framework as mesoframe
import mesocircuit.analysis.spike_analysis as sana
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
# Preprocess the data (spatial and temporal binning), and compute statistics.
# Time measurements are printed.

functions = [
    [sana.preprocess_data, [circuit]],
    [sana.compute_statistics, [circuit]],
    [sana.merge_h5_files_populations, [circuit]]
]

pt.run_parallel_functions_sequentially(functions, os.path.basename(__file__))
