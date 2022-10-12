from mesocircuit import mesocircuit_framework as mesoframe
import parametersets
import parameters as ps

################################################################################

name = 'favorite'
custom_params = parametersets.ps_dicts[name]

# custom_params = {
#    'net_dict': {
#        'N_scaling': 0.5,
#        'g': ps.ParameterRange([-4., -5.]),
#        'bg_rate': ps.ParameterRange([8., 10.])
#    }}

meso = mesoframe.MesocircuitExperiment(name, custom_params)
print(meso.parameterview)


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
