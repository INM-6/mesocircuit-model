import os
import numpy as np
import helpers
import prelim_analysis as ana

parameterview = helpers.evaluate_parameterspaces(
    filename='parameterspaces',
    with_base_params=False)

# local test run
data_path, ps_id = parameterview['local_microcircuit'][0]
if 1:
    jobscript = os.path.join(data_path, 'jobscripts', ps_id , 'network.sh')
    os.system('sh ' + jobscript)

# TODO submission of jobscripts to JURECA


# preliminary analysis just for testing
if 1:
    raster_plot_interval = np.array([500., 700.])
    firing_rates_interval = np.array([500., 1500.])
    param_path = os.path.join(data_path, 'parameters', ps_id)
    ana.evaluate(param_path, raster_plot_interval, firing_rates_interval)