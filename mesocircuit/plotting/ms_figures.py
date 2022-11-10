import mesocircuit.plotting.plotting as plotting
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pickle
import matplotlib
matplotlib.use('Agg')


def overview_and_parameters(data_dir, ref_circuit, ups_circuit):
    """
    """
    # instantiate plotting object with referene circuit
    plot = plotting.Plotting(ref_circuit)

    print('Plotting overview and parameters.')
    fig = plt.figure(figsize=(plot.plot_dict['fig_width_2col'], 5))
    gs = gridspec.GridSpec(2, 10)
    gs.update(left=0.01, right=0.96, bottom=0.09,
              top=0.94, hspace=0.5, wspace=0.5)

    titles = ['reference model,\n' + r'1 mm$^2$',
              'upscaled model,\n' + r'4$\times$4 mm$^2$']
    pad = 10  # for panel titels

    # model icons
    mc = plt.imread('microcircuit.png')
    ax = plt.subplot(gs[0, :2])
    ax.imshow(mc)
    plt.axis('off')
    ax.set_title(titles[0], pad=11)

    plot.plot_mesocircuit_icon(gs[1, :2])
    plt.gca().set_title(titles[1], pad=20)

    lims = {}
    quantities = ['full_num_neurons', 'full_indegrees', 'full_ext_indegrees']
    for q in quantities:
        lims[q] = {}
        lims[q]['vmin'] = np.min(
            [np.min(ref_circuit.net_dict[q]), np.min(ups_circuit.net_dict[q])])
        lims[q]['vmax'] = np.max(
            [np.max(ref_circuit.net_dict[q]), np.max(ups_circuit.net_dict[q])])

    for i, circuit in enumerate([ref_circuit, ups_circuit]):
        ax = plt.subplot(gs[i, 3])
        q = 'full_num_neurons'
        # two separate color bars are used because of large difference
        plot.plot_parameters_vector(
            ax,
            data=circuit.net_dict[q],
            show_num='all')
        ax.set_title('number of neurons', pad=pad)

        ax = plt.subplot(gs[i, 5:8])
        q = 'full_indegrees'
        plot.plot_parameters_matrix(
            ax,
            data=circuit.net_dict[q],
            show_num='all',
            set_bad=[0],
            vmin=lims[q]['vmin'],
            vmax=lims[q]['vmax'])
        ax.set_title('in-degree', pad=pad)

        ax = plt.subplot(gs[i, 9])
        q = 'full_ext_indegrees'
        plot.plot_parameters_vector(
            ax,
            data=circuit.net_dict[q],
            show_num='all',
            vmin=lims[q]['vmin'],
            vmax=lims[q]['vmax'])
        ax.set_title('external in-degree', pad=10)

    # TODO modify and use savefig
    plt.savefig(os.path.join(data_dir, 'overview_and_parameters.pdf'))

    return


def reference_vs_upscaled(data_dir, ref_circuit, ups_circuit):
    """
    """
    d = {}
    for i, circuit in enumerate([ref_circuit, ups_circuit]):
        if i == 0:
            prefix = 'ref'
        elif i == 1:
            prefix = 'ups'

        # load data
        for all_datatype in ['all_sptrains', 'all_pos_sorting_arrays',
                             'all_FRs', 'all_LVs', 'all_CCs_distances',
                             'all_PSDs']:
            fn = os.path.join(
                circuit.data_dir_circuit,
                'processed_data',
                all_datatype + '.h5')
            data = h5py.File(fn, 'r')
            d.update({prefix + '_' + all_datatype: data})

    # instantiate plotting object with referene circuit
    plot = plotting.Plotting(ref_circuit)

    #####

    print('Plotting rasters.')
    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 4.))
    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.12, right=0.97, bottom=0.08, top=0.9)

    labels = ['A', 'B']
    titles = ['reference model,\n' + r'1 mm$^2$',
              'upscaled model,\n' + r'1 mm$^2$ sampled']

    for i, prefix in enumerate(['ref', 'ups']):
        ax = plot.plot_raster(
            gs[0, i],
            populations=plot.Y,
            all_sptrains=d[prefix + '_all_sptrains'],
            all_pos_sorting_arrays=d[prefix + '_all_pos_sorting_arrays'],
            time_step=plot.sim_dict['sim_resolution'],
            time_interval=plot.plot_dict['raster_time_interval_short'],
            sample_step=1)
        plot.add_label(ax, labels[i])
        ax.set_title(titles[i])

        if i == 1:
            ax.set_yticklabels([])

    # TODO modify and use savefig
    plt.savefig(os.path.join(data_dir, 'ref_vs_ups_rasters.pdf'))

    #####

    print('Plotting statistics.')
    fig = plt.figure(figsize=(plot.plot_dict['fig_width_2col'], 6))
    gs = gridspec.GridSpec(2, 1)
    gs.update(left=0.07, right=0.99, bottom=0.08, top=0.93, hspace=0.5)

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
            d[prefix + '_all_PSDs'])
        for l, label in enumerate(labels[i]):
            plot.add_label(axes[l], label)
        axes[4].set_title(titles[i], pad=15)

    # TODO modify and use savefig
    plt.savefig(os.path.join(data_dir, 'rev_vs_ups_statistics.pdf'))
    return


def evoked_activity(data_dir, circuit):
    """
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

    # instantiate plotting
    plot = plotting.Plotting(circuit)

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_2col'], 7.))
    gs = gridspec.GridSpec(3, 3)
    gs.update(left=0.06, right=0.91, bottom=0.05,
              top=0.96, wspace=0.35, hspace=0.3)

    labels = ['A', 'B', 'C', 'D']

    # raster
    ax = plot.plot_raster(
        gs[:2, 0],
        plot.Y,
        d['all_sptrains'],
        d['all_pos_sorting_arrays'],
        plot.sim_dict['sim_resolution'],
        time_interval=plot.plot_dict['raster_time_interval_short'],
        sample_step=100,
        axvline=plot.net_dict['th_start'])
    ax.set_title('TC', pad=0.2, fontsize=plt.rcParams['font.size'])
    plot.add_label(ax, labels[0])

    # instantaneous firing rates
    ax = plot.plot_population_panels(
        gs[:2, 1],
        plotfunc=plot.plotfunc_instantaneous_rates,
        populations=plot.X,
        xlabel='time (ms)',
        ylabel=r'$\nu (s^{-1})$',
        sptrains=d['all_sptrains_bintime'],
        time_step=plot.ana_dict['binsize_time'],
        time_interval=plot.plot_dict['raster_time_interval_short'])
    plot.add_label(ax, labels[1])

    # spatial snapshots
    ax = plot.plot_spatial_snapshots(
        gs[2, :],
        plot.X,
        d['all_inst_rates_bintime_binspace'],
        plot.ana_dict['binsize_time'],
        orientation='horizontal',
        cbar_orientation='vertical',
        cbar_left=0.855,
        cbar_width=0.01)
    plot.add_label(ax, labels[2])

    # cross-correlation functions for thalamic pulses
    ax = plot.plot_crosscorrelation_funcs_thalamic_pulses(
        gs[:2, 2],
        plot.Y,
        d['all_CCfuncs_thalamic_pulses'],
        cbar_orientation='vertical',
        cbar_left=0.11,
        cbar_width=0.01
    )
    plot.add_label(ax, labels[3])

    plt.savefig(os.path.join(data_dir, 'evoked_activity.pdf'))
    return
