from parameterspaces import ps_dicts as custom_ps_dicts
import os
import numpy as np
import core.helpers.mesocircuit_framework as mesoframe

#key = 'microcircuit'
#key = 'local_microcircuit'
key = 'local_mesocircuit'
#key = 'local_macaqueV1'
#key = 'local_mesomacaqueV1'

parameterview = mesoframe.evaluate_parameterspaces(
    custom_ps_dicts=custom_ps_dicts,
    paramspace_keys=[key])

# theory is not fully integrated because of dependence on lif_meanfield_tools
if 0:
    mesoframe.run_parametersets(
        func=mesoframe.run_single_lmt,
        parameterview=parameterview)

mesoframe.run_parametersets(
    func=mesoframe.run_single_jobs,
    parameterview=parameterview,
    jobs=[
        'network',
        'analysis_and_plotting',
        # 'analysis',
        # 'plotting',
    ],
    machine='local')  # 'local' or 'hpc'