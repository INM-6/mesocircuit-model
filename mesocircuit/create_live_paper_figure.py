import core.helpers.mesocircuit_framework as mesoframe
import parametersets


def create_live_paper_figure(
        data_dir,
        key='favorite_evoked',
        run_parametersets=0,  # get data (run simulation network and analysis)
        run_figures=1):

    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=parametersets.ps_dicts,
        paramspace_keys=[key],
        with_base_params=False,
        data_dir=data_dir)
    print(parameterview)

    if run_parametersets:
        mesoframe.run_parametersets(
            func=mesoframe.run_single_jobs,
            parameterview=parameterview,
            jobs=[
                'network',
                'analysis_and_plotting',
            ],
            machine='hpc',
            data_dir=data_dir
        )

    if run_figures:
        # other_figures imports the Plotting class which initializes MPI,
        # therefore it cannot be loaded before run_parametersets() is executed
        # because this one launches run_*.py with MPI
        import core.plotting.other_figures as other_figures
        other_figures.live_paper(
            data_dir, key, parameterview)

    return


if __name__ == '__main__':

    data_dir = 'live_paper_figure'

    create_live_paper_figure(data_dir=data_dir)
