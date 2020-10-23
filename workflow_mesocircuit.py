import os
import numpy as np
import parameterization.helpers as helpers

key = 'microcircuit'
#key = 'local_microcircuit'
#key = 'local_mesocircuit'

parameterview = helpers.evaluate_parameterspaces(filename='parameterspaces',
    paramspace_keys = [key])

helpers.run_jobs(
    parameterview,
    jobscripts=[
        'network.sh',
        'analysis_and_plotting.sh',
#        'analysis.sh',
#        'plotting.sh',
        ],
#    run_type='run_locally',
    run_type='submit_jureca',
    )


################################################################################

# preliminary analysis from microcircuit model implementation just for testing
if 0:
    import analysis.prelim_analysis as ana
    raster_plot_interval = np.array([500., 700.])
    firing_rates_interval = np.array([500., 1500.])
    param_path = os.path.join(data_path, 'parameters', ps_id)
    ana.evaluate(param_path, raster_plot_interval, firing_rates_interval)