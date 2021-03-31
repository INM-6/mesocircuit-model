import parameters as ps
from parametersets import ps_dicts
import os
import numpy as np
import core.helpers.mesocircuit_framework as mesoframe
import core.plotting.paramspace_figure as psfig

# new parameter space key
key = 'local_mesocircuit_g-bg_rate'

# extend existing parameter dictionaries with new parameter space
custom_ps_dicts = mesoframe.extend_existing_parameterspaces(
    custom_key=key,
    custom_params={
        'net_dict': {
            'g': ps.ParameterRange([-4., -4.5]),
            'bg_rate': ps.ParameterRange([8., 8.5, 9.])
        }},
    base_key='local_mesocircuit',
    base_ps_dicts = ps_dicts)
print(f'Custom parameters of {key}:\n', custom_ps_dicts[key])

if 1:
    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=custom_ps_dicts,
        paramspace_keys=[key])

if 1:
    mesoframe.run_parametersets(
        func=mesoframe.run_single_jobs,
        parameterview=parameterview,
        jobs=[
            'network',
            'analysis_and_plotting',
        ],
        machine='local')

if 1: # should be executed locally as it depends on latex
   psfig.parameterspace_overviews(
       paramspace_key=key,
       data_dir=mesoframe.auto_data_directory())

