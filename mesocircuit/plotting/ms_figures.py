"""Manuscript figures
---------------------

Figures showing parameters and network activity of the spiking network models.
"""

import mesocircuit.plotting.plotting as plot
from mesocircuit.parameterization import helpers_analysis as helpana
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import matplotlib
matplotlib.use('Agg')


def parameters(output_dir, ref_circuit, ups_circuit):
    """
    Figure comparing selected parameters of the reference and upscaled models.

    Parameters
    ----------
    output_dir
        Output directory.
    ref_circuit
        Mesocircuit instance for the reference model.
    ups_cicruit
        Mesocircuit instance for the upscaled model.
    """

    print('Plotting parameters.')
    fig = plt.figure(figsize=(ref_circuit.plot_dict['fig_width_2col'], 7))
    gs = gridspec.GridSpec(2, 7)
    gs.update(left=0.08, right=0.95, bottom=0.06,
              top=0.9, hspace=0.5, wspace=0)

    lims = {}
    quantities = ['full_num_neurons', 'full_indegrees', 'full_ext_indegrees']
    for q in quantities:
        lims[q] = {}
        lims[q]['vmin'] = np.min(
            [np.min(ref_circuit.net_dict[q]), np.min(ups_circuit.net_dict[q])])
        lims[q]['vmax'] = np.max(
            [np.max(ref_circuit.net_dict[q]), np.max(ups_circuit.net_dict[q])])

    pad = 10
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    offset = [-0.78, 0.05]
    lcnt = 0
    for i, circuit in enumerate([ref_circuit, ups_circuit]):
        ax = plt.subplot(gs[i, :1])
        q = 'full_num_neurons'
        # two separate color bars are used because of large difference
        plot.plot_parameters_vector(
            ax,
            data=circuit.net_dict[q],
            pop_labels=circuit.plot_dict['pop_labels'],
            show_num='all',
            num_fontsize_scale=1)
        ax.set_title('number of neurons', pad=pad)
        plot.add_label(ax, labels[lcnt], offset=offset)
        lcnt += 1

        ax = plt.subplot(gs[i, 2:-2])
        q = 'full_indegrees'
        plot.plot_parameters_matrix(
            ax,
            data=circuit.net_dict[q],
            pop_labels=circuit.plot_dict['pop_labels'],
            show_num='all',
            set_bad=[0],
            vmin=lims[q]['vmin'],
            vmax=lims[q]['vmax'],
            num_fontsize_scale=0.9)
        if i == 0:
            title = 'reference model'
        elif i == 1:
            title = 'upscaled model'
        ax.set_title(title + '\n\n in-degree', pad=pad)
        plot.add_label(ax, labels[lcnt], offset=[-0.01, offset[1]])
        lcnt += 1

        ax = plt.subplot(gs[i, -1:])
        q = 'full_ext_indegrees'
        plot.plot_parameters_vector(
            ax,
            data=circuit.net_dict[q],
            pop_labels=circuit.plot_dict['pop_labels'],
            show_num='all',
            vmin=lims[q]['vmin'],
            vmax=lims[q]['vmax'],
            num_fontsize_scale=1)
        ax.set_title('external in-degree', pad=pad)
        plot.add_label(ax, labels[lcnt], offset=offset)
        lcnt += 1

    plt.savefig(os.path.join(output_dir, 'parameters.pdf'))

    return


def reference_vs_upscaled(output_dir, ref_circuit, ups_circuit,
                          plot_rasters=True, plot_statistics=True):
    """
    Figures comparing the reference and the upscaled models.

    Parameters
    ----------
    output_dir
        Output directory.
    ref_circuit
        Mesocircuit instance for the reference model.
    ups_cicruit
        Mesocircuit instance for the upscaled model.
    plot_rasters
        Whether to plot rasters and instantaneous firing rates.
    plot_statistics
        Whether to plot the statistics overview.

    Use ref_circuit when accessing parameters that are the same for both
    circuits.
    """
    d = {}
    for i, circuit in enumerate([ref_circuit, ups_circuit]):
        if i == 0:
            prefix = 'ref'
        elif i == 1:
            prefix = 'ups'

        # load data
        for all_datatype in ['all_sptrains', 'all_sptrains_bintime',
                             'all_pos_sorting_arrays',
                             'all_FRs', 'all_LVs', 'all_CCs_distances',
                             'all_PSDs']:
            fn = os.path.join(
                circuit.data_dir_circuit,
                'processed_data',
                all_datatype + '.h5')
            data = h5py.File(fn, 'r')
            d.update({prefix + '_' + all_datatype: data})

    #####
    if plot_rasters:
        print('Plotting rasters and instantaneous firing rates.')
        fig = plt.figure(figsize=(ref_circuit.plot_dict['fig_width_2col'], 4.))
        gs = gridspec.GridSpec(1, 4)
        gs.update(left=0.05, right=0.98, bottom=0.08, top=0.9, wspace=0.3)

        labels = ['A', 'B', 'C', 'D']
        titles = ['reference model,\n' + r'1 mm$^2$',
                  'upscaled model,\n' + r'1 mm$^2$ sampled']

        # rasters
        for i, prefix in enumerate(['ref', 'ups']):
            ax = plot.plot_raster(
                gs[0, i],
                all_sptrains=d[prefix + '_all_sptrains'],
                all_pos_sorting_arrays=d[prefix + '_all_pos_sorting_arrays'],
                populations=ref_circuit.ana_dict['Y'],
                pop_colors=ref_circuit.plot_dict['pop_colors'],
                pop_labels=ref_circuit.plot_dict['pop_labels'],
                time_step=ref_circuit.sim_dict['sim_resolution'],
                time_interval=[1050, 1100],
                sample_step=1,
                randomize_neuronids=True)

            plot.add_label(ax, labels[i])
            ax.set_title(titles[i])
            if i == 1:
                ax.set_yticklabels([])

        # instantaneous firing rates
        for i, prefix in enumerate(['ref', 'ups']):
            ax = plot.plot_population_panels(
                gs[0, i+2],
                plotfunc=plot.plotfunc_instantaneous_rates,
                populations=ref_circuit.ana_dict['Y'],
                xlabel='time (ms)',
                ylabel=r'$FR$ (spikes/s)' if i == 0 else '',
                sptrains=d[prefix + '_all_sptrains_bintime'],
                num_neurons=ref_circuit.net_dict['num_neurons'],
                pop_colors=ref_circuit.plot_dict['pop_colors'],
                time_step=ref_circuit.ana_dict['binsize_time'],
                # one additional time step for label
                time_interval=[1050, 1100+1],
                ylim_top=11,
                yticklabels=True if i == 0 else False)

            plot.add_label(ax, labels[i+2])
            ax.set_title(titles[i])

        plt.savefig(os.path.join(output_dir, 'ref_vs_ups_rasters.pdf'))

    #####

    if plot_statistics:
        print('Plotting statistics.')
        fig = plt.figure(figsize=(ref_circuit.plot_dict['fig_width_2col'], 6))
        gs = gridspec.GridSpec(2, 1)
        gs.update(left=0.08, right=0.99, bottom=0.08, top=0.93, hspace=0.5)

        labels = [['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                  ['H', 'I', 'J', 'K', 'L', 'M', 'N']]

        titles = ['reference model, ' + r'1 mm$^2$',
                  'upscaled model, ' + r'1 mm$^2$ sampled']

        for i, prefix in enumerate(['ref', 'ups']):
            all_CCs = {}
            all_CCs_distances = d[prefix + '_all_CCs_distances']
            for X in all_CCs_distances:
                if isinstance(all_CCs_distances[X], h5py._hl.group.Group):
                    all_CCs[X] = all_CCs_distances[X]['ccs']
                else:
                    all_CCs[X] = np.array([])

            axes = plot.plot_statistics_overview(
                gs[i],
                d[prefix + '_all_FRs'],
                d[prefix + '_all_LVs'],
                all_CCs,
                d[prefix + '_all_PSDs'],
                ref_circuit.ana_dict['Y'],
                ref_circuit.plot_dict,
                ylims_boxcharts_FRs=[0., 11.],
                ylims_boxcharts_LVs=[0., 2.],
                ylims_boxcharts_CCs=[-0.01, 0.01],
                ylims_PSDs=[0.00001, 10])
            for l, label in enumerate(labels[i]):
                plot.add_label(axes[l], label)
            axes[4].set_title(titles[i], pad=15)

        plt.savefig(os.path.join(output_dir, 'rev_vs_ups_statistics.pdf'))

    return


def correlation(output_dir, circuit):
    """
    Figure of correlation structure.

    Parameters
    ----------
    output_dir
        Output directory.
    circuit
        Mesocircuit instance.
    """
    # load data
    d = {}
    for all_datatype in ['all_CCs_distances', 'all_cross_correlation_functions']:
        fn = os.path.join(
            circuit.data_dir_circuit, 'processed_data', all_datatype + '.h5')
        data = h5py.File(fn, 'r')
        d.update({all_datatype: data})

    # extract all_CCs from all_CCs_distances
    ccs_time_intervals = np.array(
        circuit.ana_dict['ccs_time_interval']).reshape(-1)
    all_CCs = {}
    for i, interval in enumerate(ccs_time_intervals):
        all_CCs[i] = {}
        for X in d['all_CCs_distances']:
            if X != 'TC':
                all_CCs[i][X] = d['all_CCs_distances'][X][f'ccs_{interval}ms']

    # bins used in distribution in [0,1]
    bins_unscaled = (np.arange(0, circuit.plot_dict['distr_num_bins'] + 1) /
                     circuit.plot_dict['distr_num_bins'])

    fig = plt.figure(figsize=(circuit.plot_dict['fig_width_2col'], 3))
    gs = gridspec.GridSpec(1, 2)
    # gs.update(left=0.08, right=0.99, bottom=0.08, top=0.93, hspace=0.5)

    # distributions of correlation coefficients for different time lags
    ax = plot.plot_population_panels_2cols(
        gs[0, 0],
        plotfunc=plot.plotfunc_distributions,
        populations=circuit.net_dict['populations'][:-1],
        layer_labels=circuit.plot_dict['layer_labels'],
        data2d=all_CCs,
        pop_colors=circuit.plot_dict['pop_colors'],
        xlabel='$CC$',
        ylabel='p (a.u.)',
        bins=2. * (bins_unscaled - 0.5) * 0.15,  # range adjusted
        MaxNLocatorNBins=2)

    plot.add_label(ax, 'A')

    # legend
    num = len(circuit.ana_dict['ccs_time_interval'])
    legend_labels = [
        r'$\Delta t_{CC}=$' + f'{t} ms' for t in circuit.ana_dict['ccs_time_interval']]

    colors = [plot.adjust_lightness(
        circuit.plot_dict['pop_colors'][0], 1-j/(num-1)) for j in np.arange(num)]

    lines = [matplotlib.lines.Line2D([0], [0], color=c) for c in colors]

    ax.legend(lines, legend_labels)

    #####

    # spike train cross-correlation functions
    ax = plot.plot_cross_correlation_functions(
        gs[0, 1],
        populations=circuit.net_dict['populations'][:-1],
        layer_labels=circuit.plot_dict['layer_labels'],
        all_cross_correlation_functions=d['all_cross_correlation_functions'],
        pop_colors=circuit.plot_dict['pop_colors'])

    plot.add_label(ax, 'B')

    plt.savefig(os.path.join(output_dir, 'correlation.pdf'))

    return


def evoked_activity(output_dir, circuit):
    """
    Figure showing evoked activity following a thalamocortical stimulation.

    Parameters
    ----------
    output_dir
        Output directory.
    circuit
        Mesocircuit instance.
    """
    print('Plotting evoked activity')

    # load data
    d = {}
    for all_datatype in ['all_sptrains', 'all_pos_sorting_arrays',
                         'all_sptrains_bintime',
                         'all_inst_rates_bintime_binspace',
                         'all_CCfuncs_thalamic_pulses'
                         ]:
        fn = os.path.join(
            circuit.data_dir_circuit, 'processed_data', all_datatype + '.h5')
        data = h5py.File(fn, 'r')
        d.update({all_datatype: data})

    fig = plt.figure(figsize=(circuit.plot_dict['fig_width_2col'], 7.))
    gs = gridspec.GridSpec(3, 3)
    gs.update(left=0.06, right=0.91, bottom=0.05,
              top=0.96, wspace=0.35, hspace=0.3)

    labels = ['A', 'B', 'C', 'D']

    # spatial bins
    space_bins = helpana.get_space_bins(
        circuit.net_dict['extent'], circuit.ana_dict['binsize_space'])

    # spatial snapshots
    ax = plot.plot_spatial_snapshots(
        gs[0, :],
        d['all_inst_rates_bintime_binspace'],
        circuit.ana_dict['X'],
        circuit.ana_dict['binsize_time'],
        space_bins,
        circuit.plot_dict['pop_labels'],
        circuit.plot_dict['snapshots_max_rate'],
        orientation='horizontal',
        start_time=1100.,  # ms
        cbar_orientation='vertical',
        cbar_size='2%',
        cbar_pad=0.1)
    plot.add_label(ax, labels[0])

    # raster
    ax = plot.plot_raster(
        gs[1:, 0],
        d['all_sptrains'],
        d['all_pos_sorting_arrays'],
        circuit.ana_dict['Y'],
        circuit.plot_dict['pop_colors'],
        circuit.plot_dict['pop_labels'],
        circuit.sim_dict['sim_resolution'],
        time_interval=[1050, 1150],
        sample_step=25,
        axvline=1100)
    ax.set_title('TC', pad=0.2, fontsize=plt.rcParams['font.size'])
    plot.add_label(ax, labels[1])

    # instantaneous firing rates
    ax = plot.plot_population_panels(
        gs[1:, 1],
        plotfunc=plot.plotfunc_instantaneous_rates,
        populations=circuit.ana_dict['X'],
        xlabel='time (ms)',
        ylabel=r'$FR$ (spikes/s)',
        sptrains=d['all_sptrains_bintime'],
        num_neurons=circuit.net_dict['num_neurons'],
        pop_colors=circuit.plot_dict['pop_colors'],
        time_step=circuit.ana_dict['binsize_time'],
        time_interval=[1050, 1150])
    plot.add_label(ax, labels[2])

    # cross-correlation functions for thalamic pulses
    ax = plot.plot_crosscorrelation_funcs_thalamic_pulses(
        gs[1:, 2],
        d['all_CCfuncs_thalamic_pulses'],
        circuit.ana_dict['Y'],
        circuit.net_dict['extent'],
        circuit.net_dict['th_radius'],
        circuit.plot_dict['layer_labels'],
        cbar_orientation='vertical',
        cbar_left=0.11,
        cbar_width=0.01,
        fit_speed=True)
    plot.add_label(ax, labels[3])

    plt.savefig(os.path.join(output_dir, 'evoked_activity.pdf'))
    return
