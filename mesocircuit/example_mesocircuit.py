import core.helpers.mesocircuit_framework as mesoframe

# minimal example to run base mesocircuit on hpc system
if 1:
    parameterview = mesoframe.evaluate_parameterspaces(with_base_params=True)

if 1:
    mesoframe.run_parametersets(
        func=mesoframe.run_single_jobs,
        parameterview=parameterview,
        jobs=[
            'network',
            'analysis_and_plotting',
        ],
        machine='hpc')
