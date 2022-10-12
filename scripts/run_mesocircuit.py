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

meso_exp = mesoframe.MesocircuitExperiment(name, custom_params)
print(meso_exp.parameterview)
print(meso_exp.circuits)


# TODO run_jobs could be general, need to find a good way for initializing a Mesocircuit from run_ functions

for circuit in meso_exp.circuits:
    circuit.run_jobs(
        jobs=[
            'network',
            'analysis_and_plotting',
            # 'lfp_simulation',
            # 'lfp_postprocess',
            # 'lfp_plotting',
        ],
        # machine='hpc',
        machine='local'
    )
