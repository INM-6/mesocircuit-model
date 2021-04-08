import core.helpers.mesocircuit_framework as mesoframe
import parametersets

# minimal example to run base mesocircuit on hpc system
if 1:
    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=parametersets.ps_dicts,
        paramspace_keys=['local_mesocircuit'],
        with_base_params=False)

if 1:
    mesoframe.run_parametersets(
        func=mesoframe.run_single_jobs,
        parameterview=parameterview,
        jobs=[
            'network',
            'analysis_and_plotting',
        ],
        machine='local',
        # machine='hpc'
        )
