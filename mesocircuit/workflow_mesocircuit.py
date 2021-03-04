from parameterspaces import ps_dicts as custom_ps_dicts
import os
import numpy as np
import core.helpers.mesocircuit_framework as mesoframe

#key = 'microcircuit'
#key = 'local_microcircuit'
key = 'local_mesocircuit'
#key = 'local_macaqueV1'
#key = 'local_mesomacaqueV1'

mesoframe.evaluate_parameterspaces(
    custom_ps_dicts=custom_ps_dicts,
    paramspace_keys=[key])

# theory is not fully integrated because of dependence on lif_meanfield_tools
if True:
    import run_theory
    run_theory.run_lmt(
        paramspace_keys=[key])

mesoframe.run_jobs(
    paramspace_keys=[key],
    jobs=[
        'network',
        'analysis_and_plotting',
        # 'analysis',
        # 'plotting',
    ],
    machine='local')  # 'local' or 'hpc'
