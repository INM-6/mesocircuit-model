import os
import numpy as np
import core.helpers.mesocircuit_framework as mesoframe

#key = 'microcircuit'
#key = 'local_microcircuit'
key = 'local_mesocircuit'
#key = 'local_macaqueV1'
#key = 'local_mesomacaqueV1'


from parameterspaces import ps_dicts as custom_ps_dicts
parameterview = mesoframe.evaluate_parameterspaces(
    custom_ps_dicts=custom_ps_dicts,
    paramspace_keys = [key]
    )

mesoframe.run_jobs(
    parameterview,
    jobscripts=[
        #'network.sh',
        #'analysis_and_plotting.sh',
        #'analysis.sh',
        #'plotting.sh',
        #'lfp.sh',
        ],
    run_type='run_locally',
#    run_type='submit_jureca',
    )
