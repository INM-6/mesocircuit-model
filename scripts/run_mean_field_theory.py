"""PyNEST Mesocircuit: Run Theory
---------------------------------
"""

import os
import numpy as np
from mesocircuit import mesocircuit_framework as mesoframe
import parametersets
import nnmt
import mesocircuit.plotting.plotting as plotting
import mesocircuit.plotting.figures as figures

name = 'microcircuit_MAMV1'  # reference model
# name = 'microcircuit_PD' # Potjans & Diesmann (2014) microcircuit
# name = 'mesocircuit_MAMV1' # upscaled model, does not work well
custom_params = parametersets.ps_dicts[name]
meso_exp = mesoframe.MesocircuitExperiment(name, custom_params)
circuit = meso_exp.circuits[0]

# nnmt network object of type Microcircuit
nw = nnmt.models.Microcircuit(
    network_params=os.path.join(
        circuit.data_dir_circuit, 'parameters', 'nnmt_dict.yaml'),
    analysis_params='nnmt_analysis_params.yaml')

# working point for exponentially shaped post synaptic currents
wp = nnmt.lif.exp.working_point(nw)
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

# overview figure
pl = plotting.Plotting(circuit)
figures.theory_overview(
    plot=pl,
    working_point=wp,
    frequencies=freqs,
    power=power,
    sensitivity=sensitivity_dict)
