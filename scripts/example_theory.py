import core.helpers.mesocircuit_framework as mesoframe

parameterview = mesoframe.evaluate_parameterspaces(
    with_base_params=True)

# theory is not fully integrated because of dependence on lif_meanfield_tools
mesoframe.run_parametersets(
    func=mesoframe.run_single_lmt,
    parameterview=parameterview)
