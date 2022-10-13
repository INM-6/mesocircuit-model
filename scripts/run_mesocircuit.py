from mesocircuit import mesocircuit_framework as mesoframe
import parametersets
import parameters as ps

################################################################################

#name = 'favorite'
name = 'local_mesocircuit'
custom_params = parametersets.ps_dicts[name]

meso_exp = mesoframe.MesocircuitExperiment(name, custom_params)
#meso_exp = mesoframe.MesocircuitExperiment(name, load=True)

print(meso_exp.parameterview)
print(meso_exp.circuits)

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
