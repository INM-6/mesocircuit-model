"""Simulations for manuscript figures
-------------------------------------

This script needs to be executed before `ms_figures_plotting.py`.
"""

import numpy as np
from mesocircuit import mesocircuit_framework as mesoframe
import parametersets
import ms_figures_cross_correlation_function as ccfunc

################################################################################
# Decide which model to setup and whether to submit jobs.
# 1: reference (1mm2 model of macaque V1)
# 2: upscaled_1mm2 (full upscaled model simulated but only center 1mm2 analyzed)
# 3: evoked (full upscaled model with evoked activity by thalamocortical
# stimulation)
# 4: upscaled_CCs_only (full upscaled model simulated but only analysis for
# correlation coefficients)
#
# To calculate pairwise spike train cross-correlation functions for model 2
# (after the respective network simulation and analysis has taken place),
# select: model = 2, run_jobs = False, run_ccfct = True.

model = 4
run_jobs = True
run_ccfunc = False

################################################################################
# Configure the parameters of the simulation experiments.
# The biological model time is in general set to 15 min, but for the evoked model
# only to 10 s.

t_sim = 15 * 60 * 1000.
t_sim_evoked = 10 * 1000.

if model == 1:
    name_reference = 'reference'
    custom_params_reference = dict(
        parametersets.ps_dicts['microcircuit_MAMV1'])
    custom_params_reference.update({'sim_dict': {'t_sim': t_sim}})
    custom_params_reference['sys_dict']['hpc']['network'].update(
        {'wall_clock_time': '05:00:00'}),
    custom_params_reference['sys_dict']['hpc'].update(
        {'analysis_and_plotting': {'wall_clock_time': '05:00:00'}})

if model == 2:
    name_upscaled_1mm2 = 'upscaled_1mm2'
    custom_params_upscaled_1mm2 = dict(
        parametersets.ps_dicts['mesocircuit_MAMV1'])
    custom_params_upscaled_1mm2.update({'ana_dict': {'extract_1mm2': True,
                                        'ccs_time_interval': [2., 50., 200.]}})
    custom_params_upscaled_1mm2.update({'sim_dict': {'t_sim': t_sim}})
    custom_params_upscaled_1mm2['sys_dict']['hpc']['network'].update(
        {'wall_clock_time': '05:00:00'}),
    custom_params_upscaled_1mm2['sys_dict']['hpc'].update(
        {'analysis_and_plotting': {'wall_clock_time': '05:00:00'}})

if model == 3:
    name_evoked = 'evoked'
    custom_params_evoked = dict(
        parametersets.ps_dicts['mesocircuit_MAMV1_evoked'])
    custom_params_evoked.update({'sim_dict': {'t_sim': t_sim_evoked}})
    custom_params_evoked['sys_dict']['hpc'].update(
        {'analysis_and_plotting': {'wall_clock_time': '05:00:00'}})

if model == 4:
    name_upscaled_CCs_only = 'upscaled_CCs_only'
    custom_params_upscaled_CCs_only = dict(
        parametersets.ps_dicts['mesocircuit_MAMV1'])
    custom_params_upscaled_CCs_only.update({
        'ana_dict': {'datatypes_preprocess': np.array(['positions', 'sptrains']),
                     'datatypes_statistics': np.array(['CCs_distances'])}})
    custom_params_upscaled_CCs_only.update({'sim_dict': {'t_sim': t_sim}})
    custom_params_upscaled_CCs_only['sys_dict']['hpc']['network'].update(
        {'wall_clock_time': '05:00:00'}),
    custom_params_upscaled_CCs_only['sys_dict']['hpc'].update(
        {'analysis_and_plotting': {'wall_clock_time': '05:00:00'}})

################################################################################
# Initialize MesocircuitExperiments for each parameter combination and inspect
# the custom parameters in each case.

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

if model == 4:
    meso_exp_upscaled_CCs_only = mesoframe.MesocircuitExperiment(
        name_upscaled_CCs_only, custom_params_upscaled_CCs_only)
    print(meso_exp_upscaled_CCs_only.parameterview)

################################################################################
# Submit jobs.
# For the long simulation times, we only do the analysis but not the generic
# plotting routines.

if model == 1:
    circuit = meso_exp_reference.circuits[0]

if model == 2:
    circuit = meso_exp_upscaled_1mm2.circuits[0]

if model == 3:
    circuit = meso_exp_evoked.circuits[0]

if model == 4:
    circuit = meso_exp_upscaled_CCs_only.circuits[0]

if run_jobs:
    circuit.run_jobs(
        jobs=[
            'network',
            'analysis',
            # 'plotting',
            # 'analysis_and_plotting'
        ],
        machine='hpc')

################################################################################
# Submit job for calculating and plotting pairwise spike train cross-correlation
# functions.

if model == 2 and run_jobs == False and run_ccfunc == True:
    ccfunc.write_jobscripts(circuit)
    ccfunc.run_job(circuit, machine='hpc')
