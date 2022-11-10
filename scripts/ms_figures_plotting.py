"""Plot  manuscript figures
---------------------------

This script needs to be executed after `ms_figures_simulations.py`.
"""


from mesocircuit import mesocircuit_framework as mesoframe
from mesocircuit.plotting import ms_figures as figures


################################################################################
# Load MesocircuitExperiments and pass the respective circuits to the functions
# for generating figures.

data_dir = 'ms_figures'

if 0:
    meso_exp_reference = mesoframe.MesocircuitExperiment(
        'reference', load=True)
    meso_exp_upscaled = mesoframe.MesocircuitExperiment(
        'upscaled', load=True)
    figures.overview_and_parameters(data_dir, meso_exp_upscaled.circuits[0])

if 0:
    meso_exp_reference = mesoframe.MesocircuitExperiment(
        'reference', load=True)
    meso_exp_upscaled_1mm2 = mesoframe.MesocircuitExperiment(
        'upscaled_1mm2', load=True)
    figures.reference_vs_upscaled(
        data_dir,
        meso_exp_reference.circuits[0], meso_exp_upscaled_1mm2.circuits[0])

if 0:
    meso_exp_evoked = mesoframe.MesocircuitExperiment(
        'evoked', load=True)
    figures.evoked_activity(data_dir, meso_exp_evoked.circuits[0])
