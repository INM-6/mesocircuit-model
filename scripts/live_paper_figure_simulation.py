"""Run simulation of mesocircuit with evoked activity for live paper
--------------------------------------------------------------------

This script needs to be executed before `live_paper_plotting.ipnb`.
"""

from mesocircuit import mesocircuit_framework as mesoframe
import parametersets

################################################################################
# Specify a name and the parameters for the upscaled model with evoked activtiy
# by thalamic stimulation.

name = 'live_paper_mesocircuit'
custom_params = parametersets.ps_dicts['mesocircuit_MAMV1_evoked']

################################################################################
# Initialize MesocircuitExperiments for each parameter combination and inspect
# the custom parameters in each case.

meso_exp = mesoframe.MesocircuitExperiment(
    name, custom_params)
print(meso_exp.parameterview)

################################################################################
# Submit jobs.

circuit = meso_exp.circuits[0]
circuit.run_jobs(
    jobs=[
        'network',
        'analysis_and_plotting'
    ],
    machine='hpc'
)
