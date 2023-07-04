"""PyNEST Mesocircuit: Run Mean-field Theory
--------------------------------------------

This example script shows a mean-field theoretical analysis using for network
parameters using
NNMT - the Neuronal Network Meanfield Toolbox (https://github.com/INM-6/nnmt).
"""

import os
import numpy as np
from mesocircuit import mesocircuit_framework as mesoframe
import parametersets
import nnmt
import mesocircuit.plotting.figures as figures

################################################################################
# We instantiate a MesocircuitExperiment with one circuit and use its parameters
# for creating an NNMT network object.

name = 'microcircuit_MAMV1'  # reference model
# name = 'microcircuit_PD'  # Potjans & Diesmann (2014) microcircuit

# upscaled model does not work well because theory does not account for space
# name = 'mesocircuit_MAMV1'
custom_params = parametersets.ps_dicts[name]
meso_exp = mesoframe.MesocircuitExperiment(name, custom_params)
circuit = meso_exp.circuits[0]

# NNMT network object of type Microcircuit
nw = nnmt.models.Microcircuit(
    network_params=os.path.join(
        circuit.data_dir_circuit, 'parameters', 'nnmt_dict.yaml'),
    analysis_params='nnmt_analysis_params.yaml')

################################################################################
# We calculate the working point, the transfer function, power spectra, and
# the sensitivity measure.

# working point for exponentially shaped post synaptic currents
wp = nnmt.lif.exp.working_point(nw)
print(wp)

# transfer function
nnmt.lif.exp.transfer_function(nw)
# delay distribution matrix
nnmt.network_properties.delay_dist_matrix(nw)
# effective connectivity matrix
nnmt.lif.exp.effective_connectivity(nw)
# power spectra
power = nnmt.lif.exp.power_spectra(nw)
freqs = nw.analysis_params['omegas'] / (2. * np.pi)

# sensitivity measure
sensitivity_dict = nnmt.lif.exp.sensitivity_measure_all_eigenmodes(nw)

################################################################################
# The results are plotted as an overview figure.

figures.theory_overview(
    circuit=circuit,
    working_point=wp,
    frequencies=freqs,
    power=power,
    sensitivity=sensitivity_dict)
