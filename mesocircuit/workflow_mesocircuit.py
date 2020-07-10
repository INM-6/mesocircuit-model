import os
import helpers

parameterview = helpers.evaluate_parameterspaces(
    filename='parameterspaces',
    paramspace_keys=['local_downscale'],
    with_base_params=False)

# local test run
if 1:
    data_path, ps_id = parameterview['local_downscale'][0]
    jobscript = os.path.join(data_path, 'jobscripts', ps_id , 'network.sh')
    os.system('sh ' + jobscript)


# TODO submission of jobscripts to JURECA