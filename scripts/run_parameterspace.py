"""PyNEST Mesocircuit: Run Parameterspace
-----------------------------------------

This example script shows how to iterate over parameter ranges and generate
comparison plots.
"""

###############################################################################
# Import parameters, the Python module used here to define parameter ranges.

from mesocircuit import mesocircuit_framework as mesoframe
from mesocircuit.plotting.paramspace_figure import parameterspace_overviews
import parametersets
import parameters as ps


###############################################################################
# Define a 2x3 parameter space for the downscaled Potjans-Diesmann microcircuit
# varying the relative inhibition and the Poisson background rate.

name = 'local_microcircuit_PD_g_vs_bg_rate'
params_key = 'local_microcircuit_PD'
custom_params = parametersets.ps_dicts[params_key]
custom_params['net_dict'].update({
    'g': ps.ParameterRange([-4., -4.5]),
    'bg_rate': ps.ParameterRange([7., 8., 9])})

###############################################################################
# Instantiate a `MesocircuitExperiment` and inspect the individual circuits of
# each parameter combination.

meso_exp = mesoframe.MesocircuitExperiment(name, custom_params)
print(meso_exp.parameterview)
print(meso_exp.circuits)

###############################################################################
# For each parameter combination, run the default simulation, analysis, and
# plotting scripts.

for circuit in meso_exp.circuits:
    circuit.run_jobs(
        jobs=[
            'network',
            'analysis_and_plotting',
        ],
        machine='local'
    )

###############################################################################
# For each default figure, create one overview figure that compares the data for
# different parameter combinations, using latex.

if 1:
    parameterspace_overviews(
        paramspace_key=name,
        data_dir=meso_exp.data_dir)
