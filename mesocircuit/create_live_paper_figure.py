import core.helpers.mesocircuit_framework as mesoframe
import parametersets
import core.plotting.other_figures as other_figures


def create_live_paper_figure(
        data_dir,
        base_key='base',
        custom_params={
            'net_dict': {'thalamic_input': 'pulses'},
        },
        machine='hpc',
        run_parametersets=1,
        run_figures=1):

    custom_ps_dicts = mesoframe.extend_existing_parameterspaces(
        custom_key=base_key,
        custom_params=custom_params,
        base_key=base_key,
        base_ps_dicts=parametersets.ps_dicts)
    print(f'Custom parameters of {base_key}:\n', custom_ps_dicts[base_key])

    parameterview = mesoframe.evaluate_parameterspaces(
        custom_ps_dicts=custom_ps_dicts,
        paramspace_keys=[base_key],
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
        other_figures.live_paper(
            data_dir, base_key, parameterview)

    return


if __name__ == '__main__':

    create_live_paper_figure(
        data_dir='live_paper_figure',
        base_key='local_mesocircuit',  # TODO
        machine='local',  # TODO
        run_parametersets=1,
        run_figures=1
    )
