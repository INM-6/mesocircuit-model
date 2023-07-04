"""PyNEST Mesocircuit: Figures
------------------------------

Definition of figures plotted with Plotting class in plotting.py.
"""

import mesocircuit.plotting.plotting as plot
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')


def parameters(circuit):
    """
    Creates a figure with important parameter matrices and vectors.

    Parameters
    ----------
    circuit
        A mesocircuit.Mesocircuit object with loaded parameters.
    """
    orig_fontsize = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': orig_fontsize * 0.5})

    fig = plt.figure(figsize=(circuit.plot_dict['fig_width_2col'], 9))
    gs = gridspec.GridSpec(4, 3)
    gs.update(left=0.06, right=0.92, bottom=0.05, top=0.95,
              wspace=0.6, hspace=0.5)

    axes = []
    if 'full_num_synapses' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['full_num_synapses'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='number of synapses',
            show_num=False,
            set_bad=[0])

    if 'full_indegrees' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['full_indegrees'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='in-degree',
            show_num='all',
            set_bad=[0])

    if 'beta' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['beta'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='beta (mm)',
            num_format='{:.3f}')

    if 'K_area_scaling' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['K_area_scaling'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='in-degree scaling',
            num_format='{:.3f}')

    if 'delay_offset_matrix' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['delay_offset_matrix'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='delay offset (ms)',
            num_format='{:.2f}')

    if 'prop_speed_matrix' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['prop_speed_matrix'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='propagation speed (mm/ms)',
            num_format='{:.2f}')

    if 'delay_lin_eff_mean' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['delay_lin_eff_mean'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='delay lin eff mean (ms)',
            num_format='{:.2f}')

    if 'delay_lin_eff_std' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['delay_lin_eff_std'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='delay lin eff std (ms)',
            num_format='{:.2f}')

    if 'full_weight_matrix_mean' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['full_weight_matrix_mean'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='mean weight (pA)',
            num_format='{:.0f}')

    if 'p0_raw' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_matrix(
            axes[-1],
            data=circuit.net_dict['p0_raw'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='not full p0_raw',
            show_num='all', num_format='{:.2f}',
            set_bad=[0])

    if 'full_num_neurons' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_vector(
            axes[-1],
            data=circuit.net_dict['full_num_neurons'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='number of neurons',
            show_num=False)

    if 'full_ext_indegrees' in circuit.net_dict:
        axes.append(plt.subplot(gs[len(axes)]))
        plot.plot_parameters_vector(
            axes[-1],
            data=circuit.net_dict['full_ext_indegrees'],
            pop_labels=circuit.plot_dict['pop_labels'],
            title='external in-degree',
            show_num='all')

    labels = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    for i, ax in enumerate(axes):
        plot.add_label(ax, labels[i])

    plot.savefig(circuit.data_dir_circuit, circuit.plot_dict['extension'],
                 'parameters')

    plt.rcParams.update({'font.size': orig_fontsize})
    return


def raster(circuit, all_sptrains, all_pos_sorting_arrays):
    """
    Creates a figure with a raster plot.

    Parameters
    ----------
    circuit
        A mesocircuit.Mesocircuit object with loaded parameters.
    all_sptrains
    all_pos_sorting_arrays
    """

    # population sizes
    N_X = [all_pos_sorting_arrays[X].size for X in circuit.ana_dict['X']]

    if circuit.net_dict['thalamic_input']:
        pops = circuit.ana_dict['X']
        num_neurons = N_X
    else:
        pops = circuit.ana_dict['Y']
        num_neurons = N_X[:-1]

    for time_interval in circuit.plot_dict['raster_time_intervals']:
        # full simulation duration
        if time_interval == 'all':
            time_interval = [
                0., circuit.sim_dict['t_presim'] + circuit.sim_dict['t_sim']]
            fig_width = circuit.plot_dict['fig_width_2col']
            target_num_dots = 80000
            left = 0.09
            right = 0.98
        else:
            fig_width = circuit.plot_dict['fig_width_1col']
            target_num_dots = 40000
            left = 0.17
            right = 0.92

        print(f'Plotting spike raster for interval: {time_interval} ms')

        # automatically compute a sample step for this figure
        if circuit.plot_dict['raster_sample_step'] == 'auto':
            # assume an average firing rate of 4 Hz to estimate the number of
            # dots if all neurons were shown
            rate_estim = 4.
            full_num_dots_estim = \
                np.diff(time_interval) * 1e-3 * \
                rate_estim * \
                np.sum(num_neurons)
            raster_sample_step = 1 + int(full_num_dots_estim / target_num_dots)
            print(
                f'  Automatically set raster_sample_step to {raster_sample_step}.')

        fig = plt.figure(figsize=(fig_width, 5.))
        gs = gridspec.GridSpec(1, 1)
        gs.update(top=0.98, bottom=0.1, left=left, right=right)
        ax = plot.plot_raster(
            gs[0, 0],
            pops,
            all_sptrains,
            all_pos_sorting_arrays,
            circuit.plot_dict['pop_colors'],
            circuit.plot_dict['pop_labels'],
            circuit.sim_dict['sim_resolution'],
            time_interval,
            raster_sample_step)

        plot.savefig(circuit.data_dir_circuit, circuit.plot_dict['extension'],
                     f'raster_{int(time_interval[0])}-{int(time_interval[1])}ms',
                     eps_conv=True)
    return


def instantaneous_firing_rates(circuit, all_sptrains_bintime):
    """
    Creates a figure with histograms of instantaneous firing rates.

    Parameters
    ----------    
    circuit
        A mesocircuit.Mesocircuit object with loaded parameters.
    all_sptrains_bintime
    """

    if circuit.net_dict['thalamic_input']:
        pops = circuit.ana_dict['X']
    else:
        pops = circuit.ana_dict['Y']

    for time_interval in circuit.plot_dict['raster_time_intervals']:
        # full simulation duration
        if time_interval == 'all':
            time_interval = [
                0., circuit.sim_dict['t_presim'] + circuit.sim_dict['t_sim']]
            fig_width = circuit.plot_dict['fig_width_2col']
            left = 0.09
            right = 0.98
        else:
            fig_width = circuit.plot_dict['fig_width_1col']
            left = 0.17
            right = 0.92

        print(
            f'Plotting instantaneous firing rates for interval: {time_interval} ms')

        fig = plt.figure(figsize=(fig_width, 5.))
        gs = gridspec.GridSpec(1, 1)
        gs.update(top=0.95, bottom=0.1, left=left, right=right)
        ax = plot.plot_population_panels(
            gs[0, 0],
            plotfunc=plot.plotfunc_instantaneous_rates,
            populations=pops,
            xlabel='time (ms)',
            ylabel=r'$FR$ (spikes/s)',
            sptrains=all_sptrains_bintime,
            num_neurons=circuit.net_dict['num_neurons'],
            pop_colors=circuit.plot_dict['pop_colors'],
            time_step=circuit.ana_dict['binsize_time'],
            time_interval=time_interval)

        plot.savefig(circuit.data_dir_circuit, circuit.plot_dict['extension'],
                     f'instantaneous_rates_{int(time_interval[0])}-{int(time_interval[1])}ms')
    return


def statistics_overview(circuit, all_FRs, all_LVs, all_CCs_distances, all_PSDs):
    """
    Creates a figure with boxplots and distributions of rates, LVs and CCs, and
    plots PSDs.

    Parameters
    ----------
    circuit
        A mesocircuit.Mesocircuit object with loaded parameters.
    plot
    all_FRs
    all_LVs
    all_CCs_distances
    all_PSDs
    """
    print('Plotting statistics overview.')

    # extract all_CCs from all_CCs_distances
    all_CCs = {}
    for X in all_CCs_distances:
        if isinstance(all_CCs_distances[X], h5py._hl.group.Group):
            all_CCs[X] = all_CCs_distances[X]['ccs']
        else:
            all_CCs[X] = np.array([])

    fig = plt.figure(figsize=(circuit.plot_dict['fig_width_2col'], 4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.98, bottom=0.15, top=0.95)
    axes = plot.plot_statistics_overview(
        gs[0], all_FRs, all_LVs, all_CCs, all_PSDs, circuit.ana_dict['Y'], circuit.plot_dict)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for i, label in enumerate(labels):
        plot.add_label(axes[i], label)

    plot.savefig(circuit.data_dir_circuit, circuit.plot_dict['extension'],
                 'statistics_overview')
    return


def corrcoef_distance(circuit, all_CCs_distances):
    """
    Creates a figure of Pearson correlation coefficients vs. distance.

    Parameters
    ----------
    circuit
        A mesocircuit.Mesocircuit object with loaded parameters.
    all_CCs_distances
    """
    print('Plotting correlation coefficients vs. distance.')

    fig = plt.figure(figsize=(circuit.plot_dict['fig_width_1col'], 5.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.98, bottom=0.09, left=0.17, right=0.98)
    ax = plot.plot_layer_panels(
        gs[0, 0],
        plotfunc=plot.plotfunc_CCs_distance,
        populations=circuit.ana_dict['Y'],
        layer_labels=circuit.plot_dict['layer_labels'],
        pop_colors=circuit.plot_dict['pop_colors'],
        data=all_CCs_distances,
        xlabel='distance (mm)',
        ylabel=r'$CC$')

    plot.savefig(circuit.data_dir_circuit, circuit.plot_dict['extension'],
                 'corrcoef_distance')
    return


def spatial_snapshots(circuit, all_inst_rates_bintime_binspace):
    """
    Creates a figure with consecutive snapshots.

    Parameters
    ----------
    circuit
        A mesocircuit.Mesocircuit object with loaded parameters.

    all_inst_rates_bintime_binspace
    """
    print('Plotting spatial snapshots.')

    if circuit.net_dict['thalamic_input']:
        pops = circuit.ana_dict['X']
    else:
        pops = circuit.ana_dict['Y']

    for start_time in circuit.plot_dict['snapshots_start_times']:
        fig = plt.figure(figsize=(circuit.plot_dict['fig_width_2col'], 3.))
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.09, right=0.97, top=1, bottom=0.1)
        ax = plot.plot_spatial_snapshots(
            gs[0, 0],
            pops,
            all_inst_rates_bintime_binspace,
            circuit.ana_dict['binsize_time'],
            circuit.ana_dict['space_bins'],
            circuit.plot_dict['pop_labels'],
            circuit.plot_dict['snapshots_max_rate'],
            orientation='horizontal',
            start_time=start_time,
            cbar_pad=0.5)

        plot.savefig(circuit.data_dir_circuit, circuit.plot_dict['extension'],
                     f'spatial_snapshots_{int(start_time)}ms')
    return


def crosscorrelation_funcs_thalamic_pulses(circuit, all_CCfuncs_thalamic_pulses):
    """
    Creates a figure with distance-dependent cross-correlation functions for
    thalamic pulses if the data exists.

    Parameters
    ----------
    circuit
        A mesocircuit.Mesocircuit object with loaded parameters.
    plot
    all_CCfuncs_thalamus_center
    """
    # only call plot function if data is not empty
    if 'cc_funcs' not in all_CCfuncs_thalamic_pulses[circuit.ana_dict['Y'][0]]:
        print(
            'Not plotting cross-correlation functions with thalamic pulses ' +
            'because all_CCfuncs_thalamic_pulses is empty.')
        return

    print('Plotting cross-correlation functions with thalamic pulses.')

    fig = plt.figure(figsize=(circuit.plot_dict['fig_width_1col'], 4.5))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.22, right=0.95, top=0.93, bottom=0.22)
    ax = plot.plot_crosscorrelation_funcs_thalamic_pulses(
        gs[0, 0],
        all_CCfuncs_thalamic_pulses,
        circuit.ana_dict['Y'],
        circuit.net_dict['extent'],
        circuit.net_dict['th_radius'],
        circuit.plot_dict['layer_labels'])

    plot.savefig(circuit.data_dir_circuit, circuit.plot_dict['extension'],
                 'crosscorrelation_funcs_thalamic_pulses')
    return


def theory_overview(circuit, working_point, frequencies, power, sensitivity):
    """
    """
    print('Plotting theory overview.')

    fig = plt.figure(figsize=(circuit.plot_dict['fig_width_2col'], 10))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.98, bottom=0.05, top=0.98)
    axes = plot.plot_theory_overview(
        gs[0], working_point, frequencies, power, sensitivity, circuit.ana_dict['Y'], circuit.plot_dict)
    labels = ['A', 'B', 'C', 'D', 'E']
    for i, label in enumerate(labels):
        plot.add_label(axes[i], label)

    plot.savefig(circuit.data_dir_circuit, circuit.plot_dict['extension'],
                 'theory_overview')
    return
