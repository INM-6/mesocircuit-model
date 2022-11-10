"""Run simulations for manuscript figures
-----------------------------------------

This script needs to be executed before `ms_figures_plotting.py`.
"""

from mesocircuit import mesocircuit_framework as mesoframe
import parametersets

################################################################################
# Specify a name and the parameters for the reference model, the upscaled model,
# and the upscaled model with evoked activtiy by thalamic stimulation.
# For the comparison with the reference model, we create a second version of the
# upscaled model, but analyze only the data within the center disk of an area of
# 1mm2.

name_reference = 'reference'
custom_params_reference = parametersets.ps_dicts['microcircuit_MAMV1']

name_upscaled = 'upscaled'
custom_params_upscaled = parametersets.ps_dicts['mesocircuit_MAMV1']

name_upscaled_1mm2 = 'upscaled_1mm2'
custom_params_upscaled_1mm2 = parametersets.ps_dicts['mesocircuit_MAMV1']
custom_params_upscaled_1mm2.update({'ana_dict': {'extract_1mm2': True}})

name_evoked = 'evoked'
custom_params_evoked = parametersets.ps_dicts['mesocircuit_MAMV1_evoked']

################################################################################
# Initialize MesocircuitExperiments for each parameter combination and inspect
# the custom parameters in each case.

meso_exp_reference = mesoframe.MesocircuitExperiment(
    name_reference, custom_params_reference)
print(meso_exp_reference.parameterview)

meso_exp_upscaled = mesoframe.MesocircuitExperiment(
    name_upscaled, custom_params_upscaled)
print(meso_exp_upscaled.parameterview)

meso_exp_upscaled_1mm2 = mesoframe.MesocircuitExperiment(
    name_upscaled_1mm2, custom_params_upscaled)
print(meso_exp_upscaled_1mm2.parameterview)

meso_exp_evoked = mesoframe.MesocircuitExperiment(
    name_evoked, custom_params_evoked)
print(meso_exp_evoked.parameterview)

################################################################################
# Submit jobs.

for meso_exp in [meso_exp_reference, meso_exp_upscaled,
                 meso_exp_upscaled_1mm2, meso_exp_evoked]:
    circuit = meso_exp.circuits[0]
    circuit.run_jobs(
        jobs=[
            'network',
            'analysis_and_plotting'
        ],
        machine='hpc'
    )
