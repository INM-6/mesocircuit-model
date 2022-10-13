"""PyNEST Mesocircuit: Run Plotting
-----------------------------------

Create plots of the network activity of the mesocircuit.
"""

###############################################################################
# Import the necessary modules.

import os
import sys
import numpy as np
import h5py
import mesocircuit.mesocircuit_framework as mesoframe
import mesocircuit.plotting.plotting as plotting
import mesocircuit.plotting.figures as figures
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
# Instantiate a Plotting object.
# Load preprocessed data and pre-computed statistics.

pl = plotting.Plotting(circuit)

d = {}
for datatype in np.append(circuit.ana_dict['datatypes_preprocess'],
                          circuit.ana_dict['datatypes_statistics']):
    all_datatype = 'all_' + datatype
    fn = os.path.join(circuit.data_dir_circuit,
                      'processed_data', all_datatype + '.h5')
    data = h5py.File(fn, 'r')
    d.update({all_datatype: data})

##########################################################################
# Plot figures and measure times.
# Time measurements are printed.

functions = [
    [figures.parameters,
     [pl]],

    [figures.raster,
     [pl, d['all_sptrains'], d['all_pos_sorting_arrays'], 'short']],

    [figures.raster,
     [pl, d['all_sptrains'], d['all_pos_sorting_arrays'], 'long']],

    [figures.instantaneous_firing_rates,
     [pl, d['all_sptrains_bintime'], 'short']],

    [figures.instantaneous_firing_rates,
     [pl, d['all_sptrains_bintime'], 'long']],

    [figures.statistics_overview,
     [pl, d['all_FRs'], d['all_LVs'], d['all_CCs_distances'], d['all_PSDs']]],

    [figures.corrcoef_distance,
     [pl, d['all_CCs_distances']]],

    [figures.spatial_snapshots,
     [pl, d['all_inst_rates_bintime_binspace']]],

    [figures.crosscorrelation_funcs_thalamic_pulses,
     [pl, d['all_CCfuncs_thalamic_pulses']]],
]

pt.run_serial_functions_in_parallel(functions, os.path.basename(__file__))
