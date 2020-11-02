"""PyNEST Mesocircuit: Figures
------------------------------

Definition of figures plotted with Plotting class in plotting.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..helpers.time_measurement import timeit_serial

@timeit_serial
def raster(plot, all_sptrains, all_pos_sorting_arrays, **kwargs):
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
        print('  Automatically set raster_sample_step to ' + 
                str(raster_sample_step) + '.')

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 5.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.98, bottom=0.1, left=0.17, right=0.92)
    ax = plot.plot_raster(
        gs[0,0],
        plot.X,
        all_sptrains,
        all_pos_sorting_arrays,
        plot.sim_dict['sim_resolution'],
        plot.plot_dict['raster_time_interval'],
        raster_sample_step)

    plot.savefig('raster', eps_conv=True)
    return


@timeit_serial
def statistics_overview(plot, all_rates, all_LVs, all_CCs, all_PSDs, **kwargs):
    """
    Creates a figure with boxplots and distributions of rates, LVs and CCs, and
    plots PSDs.

    Parameters
    ----------
    plot
    all_raters
    all_LVs
    all_CCs
    all_PSDs
    """
    print('Plotting statistics overview.')

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_2col'], 4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.98, bottom=0.15, top=0.95)
    axes = plot.plot_statistics_overview(
        gs[0], all_rates, all_LVs, all_CCs, all_PSDs)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for i,label in enumerate(labels):
        plot.add_label(axes[i], label)

    plot.savefig('statistics_overview')
    return