import core.helpers.mesocircuit_framework as mesoframe
import parametersets
import numpy as np

key = 'mesocircuit_test'
custom_ps_dicts = mesoframe.extend_existing_parameterspaces(
    custom_key=key,
    custom_params={
        'net_dict': {
            'indegree_scaling': np.array([
                # [3, 3, 1.5],  # increase
                # [7, 7, 0.8],  # reduce
                # [5, 5, 0.8],  # reduce
                # [5, 4, 0.7],  # reducing increases firing rate of L5E
            ]),
        }},
    base_key='base',
    base_ps_dicts=parametersets.ps_dicts)

parameterview = mesoframe.evaluate_parameterspaces(
    custom_ps_dicts=custom_ps_dicts,
    paramspace_keys=['mesocircuit_test'],
    with_base_params=False)

# theory is not fully integrated because of dependence on NNMT
mesoframe.run_parametersets(
    func=mesoframe.run_single_nnmt,
    parameterview=parameterview)
