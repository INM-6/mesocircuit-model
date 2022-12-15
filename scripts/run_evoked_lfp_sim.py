"""PyNEST Mesocircuit: Run Mesocircuit
--------------------------------------

Main example script to run a simulation of the mesocircuit with NEST and 
subsequently analyze and plot the results.
The simulation of the spiking neural network can be followed by an LFP
simulation and the corresponding postprocessing and plotting.
"""

###############################################################################
# The script `mesocircuit_framework` contains the main functionality for
# parameter evalutation and job execution.
# The mesocircuit framework allows for the evaluation of parameter spaces, but
# here we only simulate one individual parameter combination.
# Several interesting parameter combinations (overwriting default values) are
# collected in a dictionary in the script `parametersets` for convenience.

from mesocircuit import mesocircuit_framework as mesoframe
import parametersets

################################################################################
# Here, we choose the parameter set `mesocircuit_MAMV1` which is the default
# model. It is an upscaled version of the microcircuit representing area V1 of
# the Multi-Area Model (Schmidt and van Albada, 2018).
# 'mesocircuit_MAMV1_evoked` applies a thalamic stimulus to the center of the
# same model.
# For local testing, `local_microcircuit_PD` and `local_mesocircuit_PD` are good
# choices. These models base on the original microcircuit
# (Potjans and Diesmann, 2014) and are downscaled for execution on a laptop.

name = 'mesocircuit_MAMV1'
# name = 'local_mesocircuit_MAMV1'

custom_params = parametersets.ps_dicts[name]


custom_params['net_dict'].update(dict(thalamic_input='pulses'))


################################################################################
# Next, we instantiate a `MesocircuitExperiment` with the custom parameters.
# The argument `name` can be chosen freely; here we just use the name of the
# parameter set.
# Upon instantiation, data directories are created, derived parameters
# calculated, and job scripts written.
# If an already existing experiment should be loaded, a class can be
# instantiated with the arguments of the existing `name` and `load=True`.

meso_exp = mesoframe.MesocircuitExperiment(name, custom_params)

################################################################################
# A `MesocircuitExperiment` provides an overview over all the parameter
# combinations it is holding (`parameterview`) and a list of all the individual
# model instances of class `Mesocircuit`` (`circuits`).

print(meso_exp.parameterview)
print(meso_exp.circuits)

################################################################################
# For each Mesocircuit jobs can finally be launched.
# All provided jobs are run one after the other. `analysis_and_plotting` is for
# convenience combined into one job, but `analysis` and `plotting` can also be
# handled as individual jobs.
# For running the full model on an HPC cluster, `machine='hpc'` is required for
# submitting batch scripts via slurm; for a local test run `machine='local'`
# should be selected.

for circuit in meso_exp.circuits:
    circuit.run_jobs(
        jobs=[
            'network',
            'analysis_and_plotting',
            'lfp_simulation',
            'lfp_postprocess',
            'lfp_plotting',
        ],
        machine='hpc',
        # machine='local'
    )
