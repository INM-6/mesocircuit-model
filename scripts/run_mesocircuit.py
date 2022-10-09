from mesocircuit import mesocircuit_framework as mesoframe
import parametersets

################################################################################
#

if 1:
    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=parametersets.ps_dicts,
        paramspace_keys=[
            'favorite',
            # 'favorite_evoked'
        ],
        with_base_params=False)


if 0:
    mesoframe.run_parametersets(
        func=mesoframe.run_single_jobs,
        parameterview=parameterview,
        jobs=[
            'network',
            'analysis_and_plotting',
            # 'lfp_simulation',
            # 'lfp_postprocess',
            # 'lfp_plotting',
        ],
        machine='hpc',
        # machine='local'
    )
