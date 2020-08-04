import os
import numpy as np
import parameterization.helpers as helpers

key = 'local_microcircuit'
#key = 'local_mesocircuit'

parameterview = helpers.evaluate_parameterspaces(filename='parameterspaces',
paramspace_keys = [key])

# local test run
data_path, ps_id = parameterview[key][0]
if 1:
    jobscript = os.path.join(data_path, 'jobscripts', ps_id , 'network.sh')
    os.system('sh ' + jobscript)

# TODO submission of jobscripts to JURECA


# preliminary analysis from microcircuit just for testing
if 0:
    import analysis.prelim_analysis as ana
    raster_plot_interval = np.array([500., 700.])
    firing_rates_interval = np.array([500., 1500.])
    param_path = os.path.join(data_path, 'parameters', ps_id)
    ana.evaluate(param_path, raster_plot_interval, firing_rates_interval)

if 1:
    jobscript = os.path.join(data_path, 'jobscripts', ps_id, 'analysis.sh')
    os.system('sh ' + jobscript)

if 1:
    jobscript = os.path.join(data_path, 'jobscripts', ps_id, 'plotting.sh')
    os.system('sh ' + jobscript)
