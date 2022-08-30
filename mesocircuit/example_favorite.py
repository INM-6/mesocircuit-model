import core.helpers.mesocircuit_framework as mesoframe
import parametersets

if 1:
    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=parametersets.ps_dicts,
        paramspace_keys=[
            'favorite',
            # 'favorite_evoked'
        ],
        with_base_params=False)

if 1:
    mesoframe.run_parametersets(
        func=mesoframe.run_single_jobs,
        parameterview=parameterview,
        jobs=[
            'network',
            'analysis_and_plotting',
            # 'lfp_simulation',
            # 'lfp_plotting',
        ],
        machine='hpc',
    )
