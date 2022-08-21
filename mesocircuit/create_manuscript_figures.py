import core.helpers.mesocircuit_framework as mesoframe
import parametersets


def create_figs_reference_vs_upscaled(
        data_dir,
        ref_model='microcircuit',
        ups_model='base',
        machine='hpc',
        run_parametersets=0,
        run_figures=1):

    # extract 1mm2 from full upscaled model
    ups_model_1mm2 = ups_model + '_1mm2'
    custom_ps_dicts = mesoframe.extend_existing_parameterspaces(
        custom_key=ups_model_1mm2,
        custom_params={'ana_dict': {'extract_1mm2': True}},
        base_key=ups_model,
        base_ps_dicts=parametersets.ps_dicts)
    print(
        f'Custom parameters of {ups_model_1mm2}:\n',
        custom_ps_dicts[ups_model_1mm2])

    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=custom_ps_dicts,
        paramspace_keys=[ref_model, ups_model_1mm2],
        with_base_params=False,
        data_dir=data_dir)
    print(f'Parameterview:\n {parameterview}')

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
        # ms_figures imports the Plotting class which initializes MPI,
        # therefore it cannot be loaded before run_parametersets() is executed
        # because this one launches run_*.py with MPI
        import core.plotting.ms_figures as ms_figures
        ms_figures.reference_vs_upscaled(
            data_dir, ref_model, ups_model_1mm2, parameterview)

    return


if __name__ == '__main__':

    data_dir = 'ms_figures'

    if 1:
        create_figs_reference_vs_upscaled(data_dir=data_dir)
    else:  # for testing locally
        create_figs_reference_vs_upscaled(
            data_dir=data_dir,
            ref_model='local_microcircuit',
            ups_model='local_mesocircuit',
            machine='local',
            run_parametersets=1,
            run_figures=1)
