import core.helpers.mesocircuit_framework as mesoframe
import parametersets


def create_live_paper_figure(
        data_dir,
        base_key='base',
        custom_key='evoked_live_paper',
        custom_params={  # TODO parameters not final
            'net_dict': {'thalamic_input': 'pulses', 'th_rel_radius': 0.1},
        },
        machine='hpc',
        run_parametersets=0,  # get data (run simulation network and analysis)
        run_figures=1):

    custom_ps_dicts = mesoframe.extend_existing_parameterspaces(
        custom_key=custom_key,
        custom_params=custom_params,
        base_key=base_key,
        base_ps_dicts=parametersets.ps_dicts)
    print(f'Custom parameters of {custom_key}:\n', custom_ps_dicts[custom_key])

    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=custom_ps_dicts,
        paramspace_keys=[custom_key],
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
            machine=machine,
            data_dir=data_dir
        )

    if run_figures:
        # other_figures imports the Plotting class which initializes MPI,
        # therefore it cannot be loaded before run_parametersets() is executed
        # because this one launches run_*.py with MPI
        import core.plotting.other_figures as other_figures
        other_figures.live_paper(
            data_dir, custom_key, parameterview)

    return


if __name__ == '__main__':

    data_dir = 'live_paper_figure'

    if 1:
        create_live_paper_figure(data_dir=data_dir)
    else:  # for testing locally
        create_live_paper_figure(
            data_dir=data_dir,
            base_key='local_mesocircuit',
            machine='local',
            run_parametersets=1,
            run_figures=1
        )
