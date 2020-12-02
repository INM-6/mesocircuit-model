"""PyNEST Mesocircuit: Figures
------------------------------

Definition of figures plotted with Plotting class in plotting.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # noqa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def raster(plot, all_sptrains, all_pos_sorting_arrays):
    """
    Creates a figure with a raster plot.

    Parameters
    ----------
    plot
    all_sptrains
    all_pos_sorting_arrays
    """
    print('Plotting spike raster.')

    # automatically compute a samatplotlib. step for this figure
    if plot.plot_dict['raster_sample_step'] == 'auto':
        target_num_dots = 40000
        # assume an average firing rate of 4 Hz to estimate the number of
        # dots if all neurons were shown
        rate_estim = 4.
        full_num_dots_estim = \
            np.diff(plot.plot_dict['raster_time_interval']) * 1e-3 * \
            rate_estim * \
            np.sum(plot.net_dict['num_neurons'])
        raster_sample_step = 1 + int(full_num_dots_estim / target_num_dots)
        print('  Automatically set raster_sample_step to ' + str(raster_sample_step) + '.')

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 5.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.98, bottom=0.1, left=0.17, right=0.92)
    ax = plot.plot_raster(
        gs[0, 0],
        plot.X,
        all_sptrains,
        all_pos_sorting_arrays,
        plot.sim_dict['sim_resolution'],
        plot.plot_dict['raster_time_interval'],
        raster_sample_step)

    plot.savefig('raster', eps_conv=True)


def statistics_overview(plot, all_FRs, all_LVs, all_CCs_distances, all_PSDs):
    """
    Creates a figure with boxplots and distributions of rates, LVs and CCs, and
    plots PSDs.

    Parameters
    ----------
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
        all_CCs[X] = all_CCs_distances[X]['ccs']

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_2col'], 4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.98, bottom=0.15, top=0.95)
    axes = plot.plot_statistics_overview(
        gs[0], all_FRs, all_LVs, all_CCs, all_PSDs)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for i, label in enumerate(labels):
        plot.add_label(axes[i], label)

    plot.savefig('statistics_overview')


def corrcoef_distance(plot, all_CCs_distances):
    """
    Creates a figure of Pearson correlation coefficients vs. distance.

    Parameters
    ----------
    plot
    all_CCs_distances
    """
    print('Plotting correlation coefficients vs. distance.')

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 5.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.98, bottom=0.09, left=0.17, right=0.98)
    ax = plot.plot_layer_panels(
         gs[0, 0],
         plotfunc=plot.plotfunc_CCs_distance,
         data=all_CCs_distances,
         xlabel='distance (mm)',
         ylabel='CC')

    plot.savefig('corrcoef_distance')


def spatial_snapshots(plot, all_inst_rates_bintime_binspace):
    """
    Creates a figure with consecutive snapshots.

    Parameters
    ----------
    plot
    all_inst_rates_bintime_binspace
    """
    print('Plotting spatial snapshots.')

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 4.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.17, right=0.97, top=0.99, bottom=0.2)
    ax = plot.plot_spatial_snapshots(
         gs[0, 0],
         plot.X,
         all_inst_rates_bintime_binspace,
         plot.ana_dict['binsize_time'],
         plot.ana_dict['binsize_space'])
    plot.savefig('spatial_snapshots')


def crosscorrelation_funcs_thalamic_pulses(plot, all_CCfuncs_thalamic_pulses):
    """
    Creates a figure with distance-dependent cross-correlation functions for
    thalamic pulses if the data exists.

    Parameters
    ----------
    plot
    all_CCfuncs_thalamus_center
    """
    # only call plot function if data is not empty
    if 'cc_funcs' not in all_CCfuncs_thalamic_pulses[plot.Y[0]]:
        print('Not plotting cross-correlation functions with thalamic pulses ' +
              'because all_CCfuncs_thalamic_pulses is empty.')
        return

    print('Plotting cross-correlation functions with thalamic pulses.')

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 4.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.22, right=0.97, top=0.99, bottom=0.22)
    ax = plot.plot_crosscorrelation_funcs_thalamic_pulses(
         gs[0, 0],
         plot.Y,
         all_CCfuncs_thalamic_pulses)
    plot.savefig('crosscorrelation_funcs_thalamic_pulses')
