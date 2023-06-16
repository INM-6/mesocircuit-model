"""Run simulations for manuscript figures
-----------------------------------------

This script needs to be executed before `ms_figures_plotting.py`.
"""

from mesocircuit import mesocircuit_framework as mesoframe
import parametersets

################################################################################
# Decide which model to setup and whether to submit jobs.
# 0: upscaled (not used for final figures)
# 1: reference
# 2: upscaled_1mm2
# 3: evoked

model = 1
run_jobs = True

################################################################################
# Configure the pararameters of the reference model, the upscaled model,
# and the upscaled model with evoked activtiy by thalamic stimulation.
# For the comparison with the reference model, we analyze only the data within
# the center disk of an area of 1 mm2 of the upscaled model.
# The biological model time is set to 5 min.

t_sim = 5 * 60 * 1000.

if model == 0:
    name_upscaled = 'upscaled'
    custom_params_upscaled = dict(parametersets.ps_dicts['mesocircuit_MAMV1'])
    custom_params_upscaled.update({'sim_dict': {'t_sim': t_sim}})
    custom_params_upscaled['sys_dict']['hpc']['network'].update(
        {'wall_clock_time': '02:00:00'}),
    custom_params_upscaled['sys_dict']['hpc'].update(
        {'analysis_and_plotting': {'wall_clock_time': '02:00:00'}})

if model == 1:
    name_reference = 'reference'
    custom_params_reference = dict(
        parametersets.ps_dicts['microcircuit_MAMV1'])
    custom_params_reference.update({'sim_dict': {'t_sim': t_sim}})
    custom_params_reference['sys_dict']['hpc']['network'].update(
        {'wall_clock_time': '02:00:00'}),
    custom_params_reference['sys_dict']['hpc'].update(
        {'analysis_and_plotting': {'wall_clock_time': '02:00:00'}})

if model == 2:
    name_upscaled_1mm2 = 'upscaled_1mm2'
    custom_params_upscaled_1mm2 = dict(
        parametersets.ps_dicts['mesocircuit_MAMV1'])
    custom_params_upscaled_1mm2.update({'ana_dict': {'extract_1mm2': True}})
    custom_params_upscaled_1mm2.update({'sim_dict': {'t_sim': t_sim}})
    custom_params_upscaled_1mm2['sys_dict']['hpc']['network'].update(
        {'wall_clock_time': '02:00:00'}),
    custom_params_upscaled_1mm2['sys_dict']['hpc'].update(
        {'analysis_and_plotting': {'wall_clock_time': '02:00:00'}})

if model == 3:
    name_evoked = 'evoked'
    custom_params_evoked = dict(
        parametersets.ps_dicts['mesocircuit_MAMV1_evoked'])
    custom_params_evoked.update({'sim_dict': {'t_sim': 10000.}})

################################################################################
# Initialize MesocircuitExperiments for each parameter combination and inspect
# the custom parameters in each case.

if model == 0:
    meso_exp_upscaled = mesoframe.MesocircuitExperiment(
        name_upscaled, custom_params_upscaled)
    print(meso_exp_upscaled.parameterview)

if model == 1:
    meso_exp_reference = mesoframe.MesocircuitExperiment(
        name_reference, custom_params_reference)
    print(meso_exp_reference.parameterview)

if model == 2:
    meso_exp_upscaled_1mm2 = mesoframe.MesocircuitExperiment(
        name_upscaled_1mm2, custom_params_upscaled_1mm2)
    print(meso_exp_upscaled_1mm2.parameterview)

if model == 3:
    meso_exp_evoked = mesoframe.MesocircuitExperiment(
        name_evoked, custom_params_evoked)
    print(meso_exp_evoked.parameterview)

################################################################################
# Submit jobs.
# For the long simulation times, we only do the analysis but not the generic
# plotting routines.

if model == 0:
    circuit = meso_exp_upscaled.circuits[0]

if model == 1:
    circuit = meso_exp_reference.circuits[0]

if model == 2:
    circuit = meso_exp_upscaled_1mm2.circuits[0]

if model == 3:
    circuit = meso_exp_evoked.circuits[0]

if run_jobs:
    circuit.run_jobs(
        jobs=[
            'network',
            'analysis',
            # 'analysis_and_plotting'
        ],
        machine='hpc'
    )
