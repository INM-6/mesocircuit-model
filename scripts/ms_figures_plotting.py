"""Plot  manuscript figures
---------------------------

This script needs to be executed after `ms_figures_simulations.py`.
"""

import os
from mesocircuit import mesocircuit_framework as mesoframe
from mesocircuit.plotting import ms_figures as figures


################################################################################
# Load MesocircuitExperiments and pass the respective circuits to the functions
# for generating figures.

output_dir = 'ms_figures'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if 0:
    meso_exp_reference = mesoframe.MesocircuitExperiment(
        'reference', load=True)
    meso_exp_upscaled = mesoframe.MesocircuitExperiment(
        'upscaled', load=True)
    figures.overview_and_parameters(
        output_dir,
        meso_exp_reference.circuits[0], meso_exp_upscaled.circuits[0])

if 0:
    meso_exp_reference = mesoframe.MesocircuitExperiment(
        'reference', load=True)
    meso_exp_upscaled_1mm2 = mesoframe.MesocircuitExperiment(
        'upscaled_1mm2', load=True)
    figures.reference_vs_upscaled(
        output_dir,
        meso_exp_reference.circuits[0], meso_exp_upscaled_1mm2.circuits[0])

if 0:
    meso_exp_evoked = mesoframe.MesocircuitExperiment(
        'evoked', load=True)
    figures.evoked_activity(output_dir, meso_exp_evoked.circuits[0])
