import core.helpers.mesocircuit_framework as mesoframe
import parametersets
import core.plotting.ms_figures as ms_figures


def create_figs_reference_vs_upscaled(
        data_dir,
        ref_model='microcircuit',
        ups_model='mesocircuit',
        machine='hpc',
        run_parametersets=1,
        run_figures=1):

    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=parametersets.ps_dicts,
        paramspace_keys=[ref_model, ups_model],
        with_base_params=False,
        data_dir=data_dir)

    if run_parametersets:
        mesoframe.run_parametersets(
            func=mesoframe.run_single_jobs,
            parameterview=parameterview,
            jobs=[
                'network',
                'analysis_and_plotting',
            ],
            machine=machine,
            data_dir=data_dir
        )

    if run_figures:
        ms_figures.reference_vs_upscaled(
            data_dir, ref_model, ups_model, parameterview)

    return


if __name__ == '__main__':

    data_dir = 'ms_figures'

    if 1:
        create_figs_reference_vs_upscaled(
            data_dir=data_dir,
            ref_model='local_mesocircuit',  # TODO microcircuit
            ups_model='local_mesocircuit',  # TODO base mesocircuit
            machine='local',
            run_parametersets=0,
            run_figures=0)
