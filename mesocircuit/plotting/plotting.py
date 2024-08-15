"""Plotting
-----------

Functions starting with 'plot_' plot to a gridspec cell and are used in
figures.py.
"""

from mesocircuit.parameterization.base_plotting_params import rcParams
from re import I
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
from mpi4py import MPI
import os
import warnings
import h5py
import numpy as np
import matplotlib
from scipy.optimize import curve_fit
from mesocircuit.helpers.io import load_h5_to_sparse_X
matplotlib.use('Agg')
matplotlib.rcParams.update(rcParams)

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


def plot_raster(
        gs,
        all_sptrains,
        all_pos_sorting_arrays,
        populations,
        pop_colors,
        pop_labels,
        time_step,
        time_interval,
        sample_step,
        xticklabels=True,
        xlabel=True,
        yticks=True,
        yticklabels=True,
        markersize_scale=0.25,
        axvline=None,
        randomize_neuronids=False,
        random_seed=567):
    """
    Plots spike raster to gridspec cell.

    Neurons are sorted according to sorting_axis applied in
    all_pos_sorting_arrays if randomize_neuronids=False.

    Parameters
    ----------
    gs
        A gridspec cell to plot into.
    all_sptrains
        Open h5 file with all spike trains.
    all_pos_sorting_arrays
        Open h5 file with position sorting arrays.
    populations
        List of population names.
    pop_colors
        Population colors.
    pop_labels
        Population labels.
    time_step
        Time step corresponding to spike trains.
    time_interval
        Time interval to plot.
    sample_step
        Every sample_step'th neuron is shown (default being 1 means that all
        neurons are shown).
    xticklabels
        Boolean indicating if x-ticklabels shall be plotted.
    xlabel
        Boolean indicating if x-label shall be plotted.
    yticklabels
        Boolean indicating if y-ticklabels shall be plotted.
    markersize_scale
        Scaling factor for marker size.
    axvhline
        Time point of vertical line in ms.
    randomize_neuronids
        Whether to scramble neuron ids.
    random_seed
        Random seed used to scramble neuron ids.

    Returns
    -------
    ax
        Axis to put a label to.
    """
    nums_shown = []
    yticks = []
    ax = plt.subplot(gs)
    for i, X in enumerate(populations):
        data = load_h5_to_sparse_X(X, all_sptrains)

        # slice according to time interval
        time_indices = np.arange(
            time_interval[0] / time_step,
            time_interval[1] / time_step).astype(int)
        data = data[:, time_indices]

        # deterministically randomize neuron ids
        if randomize_neuronids:
            np.random.seed(random_seed)
            rnd_indices = np.arange(np.shape(data)[0])
            np.random.shuffle(rnd_indices)
            data = data[rnd_indices, :]
        # sort according to spatial axis
        else:
            space_indices = all_pos_sorting_arrays[X][()]
            data = data[space_indices, :]

        # subsample if specified
        if sample_step > 1:
            sample_indices = np.zeros(data.shape[0], dtype=bool)
            sample_indices[::sample_step] = True
            data = data[sample_indices, :]

        # final number of neurons to be shown
        num_neurons = data.shape[0]

        # get x,y indices and plot
        y, x = np.nonzero(data.toarray())
        ax.plot(x * time_step + time_interval[0],
                -(np.sum(nums_shown) + y),
                marker='$.$',
                markersize=matplotlib.rcParams['lines.markersize'] *
                markersize_scale,
                color=pop_colors[i],
                markeredgecolor='none',
                linestyle='',
                rasterized=True)
        nums_shown.append(num_neurons)
        yticks.append(-np.sum(nums_shown) + 0.5 * nums_shown[-1])

    # draw lines to separate populations on top
    for i, X in enumerate(populations[:-1]):
        ax.plot(time_interval, [-np.sum(nums_shown[:i + 1])] * 2,
                'k',
                linewidth=matplotlib.rcParams['axes.linewidth'])

    if axvline:
        plt.axvline(x=axvline, color='k')

    ax.set_xlim(time_interval[0], time_interval[1])
    ax.set_ylim(-np.sum(nums_shown), 0)

    ax.set_yticks(yticks)

    if xlabel:
        ax.set_xlabel('time (ms)')
    if not xticklabels:
        ax.set_xticklabels([])
    if yticklabels:
        ax.set_yticklabels(pop_labels[:len(nums_shown)])
    else:
        ax.set_yticklabels([])
    return ax


def plot_statistics_overview(
        gs, all_FRs, all_LVs, all_CCs, all_PSDs,
        populations, plot_dict,
        ylims_boxcharts_FRs=False,
        ylims_boxcharts_LVs=False,
        ylims_boxcharts_CCs=False,
        ylims_PSDs=False):
    """
    Plots statisctis overview for firing rates, local coefficients of variation,
    correlation coefficients, and power spectral densities.

    Parameters
    ----------
    gs
        A gridspec cell to plot into.
    all_FRs
        Open h5 file with firing rates.
    all_LVs
        Open h5 file with local coefficients of variation.
    all_CCs
        Open h5 file with correlation coefficients.
    all_PSDs
        Open h5 file with power spectral densities.
    populations
        List of population names.
    plot_dict
        Dictionary with plotting parameters.
    ylims_boxcharts_FRs
        Y-limits for boxcharts showing firing rates.
    ylims_boxcharts_LVs
        Y-limits for boxcharts showing local coefficients of variation.
    ylims_boxcharts_CCs
        Y-limits for boxcharts showing correlation coefficients.
    ylims_PSDs
        Y-limits for boxcharts showing power spectral densities.

    Returns
    -------
        axes
            Axes to put labels to.
    """
    axes = [0] * 7
    gs_cols = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=gs,
                                               wspace=0.5)

    # column 0: boxcharts
    gs_c0 = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_cols[0, :2], hspace=0.5)

    # top: FRs
    print('  Plotting boxcharts: rates')
    axes[0] = plot_boxcharts(gs_c0[0, 0],
                             all_FRs,
                             populations,
                             pop_labels=plot_dict['pop_labels'],
                             pop_colors=plot_dict['pop_colors'],
                             xlabel='', ylabel=r'$FR$ (spikes/s)',
                             xticklabels=False,
                             ylims=ylims_boxcharts_FRs)

    # middle: LVs
    print('  Plotting boxcharts: LVs')
    axes[1] = plot_boxcharts(gs_c0[1, 0],
                             all_LVs,
                             populations,
                             pop_labels=plot_dict['pop_labels'],
                             pop_colors=plot_dict['pop_colors'],
                             xlabel='', ylabel=r'$LV$',
                             xticklabels=False,
                             ylims=ylims_boxcharts_LVs)

    # bottom: CCs
    print('  Plotting boxcharts: CCs')
    axes[2] = plot_boxcharts(gs_c0[2, 0],
                             all_CCs,
                             populations,
                             pop_labels=plot_dict['pop_labels'],
                             pop_colors=plot_dict['pop_colors'],
                             xlabel='', ylabel=r'$CC$',
                             ylims=ylims_boxcharts_CCs)

    # columns 1, 2, 3: distributions

    # bins used in distribution in [0,1]
    bins_unscaled = (np.arange(0, plot_dict['distr_num_bins'] + 1) /
                     plot_dict['distr_num_bins'])

    # left: FRs
    print('  Plotting distributions: FRs')
    axes[3] = plot_layer_panels(gs_cols[0, 3:5],
                                xlabel=r'$FR$ (spikes/s)',
                                plotfunc=plotfunc_distributions,
                                populations=populations,
                                layer_labels=plot_dict['layer_labels'],
                                pop_colors=plot_dict['pop_colors'],
                                bins=bins_unscaled *
                                plot_dict['distr_max_rate'],
                                data=all_FRs,
                                MaxNLocatorNBins=3,
                                ylabel='p (a.u.)')

    # middle: LVs
    print('  Plotting distributions: LVs')
    axes[4] = plot_layer_panels(gs_cols[0, 5:7],
                                xlabel=r'$LV$',
                                plotfunc=plotfunc_distributions,
                                populations=populations,
                                layer_labels=plot_dict['layer_labels'],
                                pop_colors=plot_dict['pop_colors'],
                                bins=bins_unscaled * plot_dict['distr_max_lv'],
                                data=all_LVs,
                                MaxNLocatorNBins=3)

    # right: CCs
    print('  Plotting distributions: CCs')
    axes[5] = plot_layer_panels(gs_cols[0, 7:9],
                                xlabel=r'$CC$',
                                plotfunc=plotfunc_distributions,
                                populations=populations,
                                layer_labels=plot_dict['layer_labels'],
                                pop_colors=plot_dict['pop_colors'],
                                bins=2. * (bins_unscaled - 0.5) *
                                plot_dict['distr_max_cc'],
                                data=all_CCs,
                                MaxNLocatorNBins=2)

    # column 4: PSDs
    print('  Plotting PSDs')
    axes[6] = plot_layer_panels(gs_cols[0, 10:],
                                xlabel='frequency (Hz)', ylabel=r'$PSD$ (s$^{-2}$/Hz)',
                                plotfunc=plotfunc_PSDs,
                                populations=populations,
                                layer_labels=plot_dict['layer_labels'],
                                pop_colors=plot_dict['pop_colors'],
                                psd_max_freq=plot_dict['psd_max_freq'],
                                data=all_PSDs,
                                ylims=ylims_PSDs)
    return axes


def plot_spatial_snapshots(
        gs,
        all_inst_rates_bintime_binspace,
        populations,
        binsize_time,
        space_bins,
        pop_labels,
        snapshots_max_rate,
        orientation='horizontal',
        start_time=1000.,  # ms
        step=1,  # multiplication
        nframes=30,
        tickstep=2,
        cbar=True,
        cbar_orientation='horizontal',
        cbar_size='5%',
        cbar_pad=0.4):
    """
    Plots a sequence of snapshots of spatiotemporally binned firing rates.

    Parameters
    ----------
    gs
        A gridspec cell to plot into.
    all_inst_rates_bintime_binspace
        Open h5 file with all instantaneous firing rates binned in time and
        space.
    populations
        List of population names.
    binsize_time
        Temporal bin size in ms.
    space_bins
        Spatial bins.
    orientation
        Vertical or horizontal plot.
    start_time
        Start time in ms.
    step
        Step between individual frames.
    nframes
        Number of frames to show.
    tickstep
        Step of showing ticks.
    cbar
        Whether to plot a color bar.
    cbar_orientation
        Orientation of the color bar.
    cbar_size
        Width (or height) of the color bar in %.
    cbar_pad
        Pad of the color bar.

    Returns
    -------
    ax
        Axis to put a label to.
    """

    start_frame = int(start_time / binsize_time)
    end_frame = start_frame + (nframes - 1) * step
    times = np.arange(start_frame, end_frame + 1, step) * binsize_time

    numbins = space_bins.size - 1

    # minimum rate a bit below 0 to avoid issues with cmap.set_under()
    vmin = -0.001
    # separator between sub-panels masked with cmap.set_under()
    val_sep = -1

    for X in populations:
        data = load_h5_to_sparse_X(X, all_inst_rates_bintime_binspace)
        data = data[:, start_frame:end_frame + 1:step].toarray()
        data = data.reshape((numbins, -1, data.shape[-1]))

        # append frames as columns
        separator_frames = np.array([val_sep] * (numbins)).reshape(-1, 1)
        data0 = np.concatenate((data[:, :, 0], separator_frames), axis=1)
        for n in np.arange(nframes - 1):
            data_apnd = np.concatenate((data0, data[:, :, n + 1]), axis=1)
            data0 = np.concatenate((data_apnd, separator_frames), axis=1)

        # append populations as rows
        separator_pops = np.array(
            [val_sep] * np.shape(data_apnd)[1]).reshape(1, -1)
        if X == populations[0]:
            plot_data0 = np.concatenate(
                (data_apnd, separator_pops), axis=0)
        else:
            plot_data = np.concatenate((plot_data0, data_apnd), axis=0)
            plot_data0 = np.concatenate(
                (plot_data, separator_pops), axis=0)

    # ticks dependent on number of spatial bins
    # (set up for default horizontal orientation)
    xy_ticks = [numbins / 2.]
    for t in np.arange(np.max([nframes - 1, len(populations) - 1])):
        xy_ticks.append(xy_ticks[-1] + numbins + 1.)

    xticks = xy_ticks[:nframes:tickstep]
    ticklabels = times[::tickstep]
    if (int(ticklabels[0]) == ticklabels[0] and
            int(ticklabels[1]) == ticklabels[1]):
        ticklabels = ticklabels.astype(int)

    ax = plt.subplot(gs)

    cmap = matplotlib.cm.get_cmap('Greys').copy()
    cmap.set_under(color='black')

    if orientation == 'horizontal':
        im = ax.imshow(plot_data, interpolation='nearest', cmap=cmap,
                       vmin=vmin,
                       vmax=snapshots_max_rate)
        ax.set_xticks(xticks)
        ax.set_xticklabels(ticklabels)
        ax.set_xlabel('time (ms)')

        ax.set_yticks(xy_ticks[:len(populations)])
        ax.set_yticklabels(pop_labels[:len(populations)])

    elif orientation == 'vertical':
        im = ax.imshow(plot_data.T, interpolation='nearest', cmap=cmap,
                       vmin=vmin,
                       vmax=snapshots_max_rate,
                       origin='lower')
        ax.set_yticks(xticks)
        ax.set_yticklabels(ticklabels)
        ax.set_ylabel('time (ms)')

        ax.set_xticks(xy_ticks[:len(populations)])
        ax.set_xticklabels(pop_labels[:len(populations)],
                           rotation=90)

    if cbar:
        cbar_label = r'$FR$ (spikes/s)'
        if cbar_orientation == 'horizontal':

            colorbar(ax, im, cbar_label, axis='bottom',
                     size=cbar_size,
                     pad=cbar_pad,
                     orientation='horizontal', extend='max')
        elif cbar_orientation == 'vertical':

            colorbar(ax, im, cbar_label, axis='right',
                     size=cbar_size,
                     pad=cbar_pad,
                     orientation='vertical', extend='max')

    return ax


def plot_population_panels_2cols(
        gs,
        plotfunc,
        populations,
        layer_labels,
        data2d,
        pop_colors,
        xlabel='',
        ylabel='',
        wspace=0.3,
        **kwargs):
    """
    Generic function to plot 2 columns of panels for an even number of populations.
    Multiple curves per population are possible.
    """
    ncols = int(np.floor(np.sqrt(len(populations))))
    nrows = len(populations) // ncols
    gsf = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=gs, wspace=wspace)

    for i, X in enumerate(populations):
        # select subplot
        ax = plt.subplot(gsf[i])
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')

        # iterate over 2 dimensional data
        num = len(data2d)
        for j in np.arange(num):
            colors = [adjust_lightness(c, 1-j/(num-1)) for c in pop_colors]
            plotfunc(ax, X, i, data=data2d[j],
                     pop_colors=colors, **kwargs)

        layer = layer_labels[int(i / 2.)]
        if i == 0:
            ax.set_title('E')
            ax.set_ylabel(ylabel + '\n' + layer)
            ax_label = ax
        if i % ncols == 0 and i != 0:
            ax.set_ylabel(layer)

        if i == 1:
            ax.set_title('I')

        if i % ncols > 0:
            ax.set_yticklabels([])

        if i >= len(populations) - 2:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])

    return ax_label


def plot_cross_correlation_functions(
        gs,
        layer_labels,
        all_cross_correlation_functions,
        pop_colors,
        lag_max_plot=None,
        scale_exp_plot=5,
        cc_max_plot=5):
    """
    """
    spcorrs = all_cross_correlation_functions

    # average cross-correlation functions
    spcorrs_mean = {}
    for X, Y in zip(['L23E', 'L23E', 'L23I',
                    'L4E', 'L4E', 'L4I',
                     'L5E', 'L5E', 'L5I',
                     'L6E', 'L6E', 'L6I'],
                    ['L23E', 'L23I', 'L23I',
                    'L4E', 'L4I', 'L4I',
                     'L5E', 'L5I', 'L5I',
                     'L6E', 'L6I', 'L6I']):

        spcorrs_mean[f'{X}:{Y}'] = spcorrs[f'{X}:{Y}'][()].mean(axis=0)

    # which time lags to plot
    lag_times = np.array(spcorrs['lag_times'])
    if not lag_max_plot:
        lag_max = lag_times[-1]
    else:
        lag_max = lag_max_plot
    inds = (lag_times >= -lag_max) & (lag_times <= lag_max)

    gsf = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs, hspace=0.5, wspace=0.5)

    for i, L in enumerate(['L23', 'L4', 'L5', 'L6']):
        ax = plt.subplot(gsf[i])

        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')

        for j, key in enumerate([f'{L}E:{L}E', f'{L}E:{L}I', f'{L}I:{L}I']):
            XY = key.split(':')
            if XY[0][-1] == 'E' and XY[1][-1] == 'E':
                color = pop_colors[::2][i]
            elif XY[0][-1] == 'I' and XY[1][-1] == 'I':
                color = pop_colors[1::2][i]
            else:
                color = 'k'

            if L == 'L23':
                label = 'L2/3' + XY[0][-1] + ':' + 'L2/3' + XY[1][-1]
            else:
                label = key

            ax.plot(lag_times[inds],
                    spcorrs_mean[key][inds] * 10**(scale_exp_plot),
                    color=color,
                    label=f'{XY[0][-1]}:{XY[1][-1]}')

        ax.set_ylim(-cc_max_plot, cc_max_plot)

        ax.set_title(layer_labels[i])
        ax.axhline(y=0, color="grey", ls=':')
        ax.axvline(x=0, color="grey", ls=':')
        ax.legend(frameon=False,
                  loc='center', bbox_to_anchor=(1., 0.8),
                  fontsize=matplotlib.rcParams['font.size'] * 0.8)

        if i == 0:
            ax_label = ax
        if i < 2:
            ax.set_xticklabels([])
        if i >= 2:
            ax.set_xlabel('time lag (ms)')

        ylabel = '$CC^s$'
        if scale_exp_plot != 1:
            ylabel += r' ($10^{' + f'{-scale_exp_plot}' + r'}$)'

        if i % 2 == 0:
            ax.set_ylabel(ylabel)

    return ax_label


def plot_crosscorrelation_funcs_thalamic_pulses(
        gs,
        all_CCfuncs_thalamic_pulses,
        populations,
        extent,
        th_radius,
        layer_labels,
        wspace=0.2,
        cbar=True,
        cbar_orientation='horizontal',
        cbar_left=0.42,  # vertical
        cbar_width=0.02,  # vertical
        cbar_bottom=0.12,  # horizontal
        cbar_height=0.02,  # horizontal
        fit_speed=False):
    """
    Plots cross-correlation functions with respect to thalamic pulses.

    Parameters
    ----------
    gs
        A gridspec cell to plot into.
    all_CCfuncs_thalamic_pulses
    populations
        List of population names.
    extent
        Network extent (in mm).
    th_radius
        Radius of thalamic input (in mm).
    layer_labels
        Layer labels.
    wspace
        Width space of gridspec.
    cbar
        Whether to show a color bar.
    cbar_orientation
        Orientation of the color bar.
    cbar_left
        Color bar argument for vertical orientation.
    cbar_width
        Color bar argument for vertical orientation.
    cbar_bottom
        Color bar argument for horizontal orientation.
    cbar_height
        Color bar argument for horizontal orientation.
    fit_speed
        Whether to fit the propogation speed.

    Returns
    -------
    ax_return
        Axis to put a label to.
    """
    ncols = int(np.floor(np.sqrt(len(populations))))
    nrows = len(populations) // ncols
    gsf = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=gs, wspace=wspace)

    for i, X in enumerate(populations):
        ax = plt.subplot(gsf[i])

        cmap = 'PuOr_r'
        vmax = 0.5
        vmin = -vmax
        color_fit = 'k'

        cc_func = all_CCfuncs_thalamic_pulses[X]['cc_funcs']
        distances = all_CCfuncs_thalamic_pulses[X]['distances_mm']
        lags = all_CCfuncs_thalamic_pulses[X]['lags_ms']
        dstep = distances[1] - distances[0]

        im = ax.imshow(cc_func,
                       cmap=cmap,
                       aspect='auto',
                       extent=[lags[0],
                               lags[-1],
                               distances[0] - dstep / 2.,
                               distances[-1] + dstep / 2.],
                       vmin=vmin,
                       vmax=vmax,
                       interpolation='nearest',
                       origin='lower')

        # calculate propagation speed
        if fit_speed:
            # start search at distance equal to the radius of the thalamic
            # input; the maximum distance is half the network extent
            min_distance = th_radius
            max_distance = extent / 2.

            # disregard values lower than 10% of the maximum
            threshold = 0.05 * np.max(cc_func)

            # distance as a function of time (used to fit offset_distance r0
            # and speed v)
            def linfunc(t, r0, v):
                return r0 + v * t

            # extract for each distance the time lag corresponding to the
            # largest value of the cross-correlation function
            speed_lags = []
            speed_distances = []
            for d, dist in enumerate(distances):
                if dist >= min_distance:
                    series = cc_func[d, :]
                    if np.all(series < threshold):
                        continue

                    max_idx = np.argmax(series)
                    max_val = series[max_idx]

                    if max_val > threshold:
                        speed_lags.append(lags[max_idx])
                        speed_distances.append(dist)

            popt, pcov = curve_fit(linfunc, speed_lags, speed_distances)
            offset_distance = popt[0]
            speed = popt[1]

            min_lag = (min_distance - offset_distance) / speed
            max_lag = (max_distance - offset_distance) / speed

            ax.plot([min_lag, max_lag],
                    [min_distance, max_distance],
                    '--', color=color_fit)
            ax.text(0.02, 0.98, r'$v_\mathrm{prop}$=' + f'{speed:.2f}\n     mm/ms',
                    ha='left', va='top',
                    transform=ax.transAxes,
                    fontsize=matplotlib.rcParams['font.size'] * 0.9,
                    color=color_fit)

        ax.axis(ax.axis('tight'))
        ax.plot([0, 0], [0, th_radius],
                '-', color='k')
        ax.text(0, th_radius, 'TC',
                horizontalalignment='center',
                verticalalignment='bottom')

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        layer = layer_labels[int(i / 2.)]
        if i == 0:
            ax.set_title('E')
            ax.set_ylabel('distance (mm)\n' + layer)
            ax_label = ax
        if i % ncols == 0 and i != 0:
            ax.set_ylabel(layer)

        if i == 1:
            ax.set_title('I')

        if i % ncols > 0:
            ax.set_yticklabels([])

        if i >= len(populations) - 2:
            ax.set_xlabel(r'time lag (ms)')
        else:
            ax.set_xticklabels([])

        if cbar:
            if i == len(populations) - 1:
                fig = plt.gcf()
                rect = np.array(ax.get_position().bounds)
                if cbar_orientation == 'horizontal':
                    rect[0] += 0.0  # left
                    rect[2] += 0.0  # width
                    rect[1] -= cbar_bottom  # bottom
                    rect[3] = cbar_height  # height
                    cax = fig.add_axes(rect)
                    cb = fig.colorbar(
                        im, cax=cax, orientation='horizontal')
                    cax.xaxis.set_label_position('bottom')
                elif cbar_orientation == 'vertical':
                    rect[0] += cbar_left  # left
                    rect[2] = cbar_width  # width
                    rect[1] += 0.0  # bottom
                    rect[3] += 0.0  # height
                    cax = fig.add_axes(rect)
                    cb = fig.colorbar(
                        im, cax=cax, orientation='vertical')
                cb.set_label(r'$CC^\nu$', labelpad=0.1)
    return ax_label


def plot_theory_overview(
        gs, working_point, frequencies, power, sensitivity,
        populations, plot_dict):
    """
    Overview of results from mean-field prediction using NNMT.

    Parameters
    ----------
    gs
        A gridspec cell to plot into.
    working_point
        Working point computed by NNMT.
    frequencies
        Frequencies for power spectra.
    power
        Power corresponding to frequencies.
    sensitivity
        Sensitivity measure from Bos et al. (2016).
    populations
        List of population names.
    plot_dict
        Dictionary with plotting parameters.

    Returns
    -------
    axes
        Axes to put labels to.    
    """
    axes = [0] * 5
    gs_cols = gridspec.GridSpecFromSubplotSpec(3, 12, subplot_spec=gs,
                                               wspace=0.5)

    # column 0: barcharts
    gs_c0 = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_cols[0, :2], hspace=0.5)

    # top: FRs
    print('  Plotting barcharts: rates')
    axes[0] = plot_barcharts(gs_c0[0, 0],
                             working_point['firing_rates'],
                             populations,
                             pop_labels=plot_dict['pop_labels'],
                             pop_colors=plot_dict['pop_colors'],
                             xlabel='', ylabel='FR (spikes/s)',
                             xticklabels=False)

    # middle: mean input
    print('  Plotting barcharts: mean input')
    axes[1] = plot_barcharts(gs_c0[1, 0],
                             working_point['mean_input'] * 1000.,
                             populations,
                             pop_labels=plot_dict['pop_labels'],
                             pop_colors=plot_dict['pop_colors'],
                             xlabel='', ylabel=r'$\mu$ (mV)',
                             xticklabels=False)

    # bottom: standard deviation of input
    print('  Plotting boxcharts: standard deviation of input')
    axes[2] = plot_barcharts(gs_c0[2, 0],
                             working_point['std_input'] * 1000.,
                             populations,
                             pop_labels=plot_dict['pop_labels'],
                             pop_colors=plot_dict['pop_colors'],
                             xlabel='', ylabel=r'$\sigma$ (mV)')

    # column 1: PSDs
    print('  Plotting power')
    axes[3] = plot_layer_panels(gs_cols[0, 4:6],
                                xlabel=r'$f$ (Hz)', ylabel='power ($1/s^2$)',
                                plotfunc=plotfunc_theory_power_spectra,
                                populations=populations,
                                layer_labels=plot_dict['layer_labels'],
                                pop_colors=plot_dict['pop_colors'],
                                data=[frequencies, power])

    gs_c2 = gridspec.GridSpecFromSubplotSpec(
        4, 4, subplot_spec=gs_cols[1:, :], hspace=0.3)

    # column 2: sensitivity measure
    print('  Plotting sensitivity measure')
    for ev, sens in sensitivity.items():
        k = int(ev)
        frequency = sens['critical_frequency']

        projection_amp = sens['sensitivity_amp']
        Z_amp = np.ma.masked_where(projection_amp == 0, projection_amp)
        projection_freq = sens['sensitivity_freq']
        Z_freq = np.ma.masked_where(projection_freq == 0, projection_freq)

        freq = '({} Hz)'.format(np.round(frequency).astype(int))
        row = 0 if k < 4 else 2
        col = k % 4

        ax = plt.subplot(gs_c2[row, col])
        if k == 0:
            axes[4] = ax
        xl = 'sources' if k > 3 else ''
        xtl = plot_dict['pop_labels'][:-1] if k > 3 else []
        yl = 'targets' if k % 4 == 0 else ''
        ytl = plot_dict['pop_labels'][:-1] if k % 4 == 0 else []

        plot_matrix(ax=ax,
                    data=Z_amp,
                    title=r'$\,Z^\mathrm{amp}$ ' + freq,
                    xlabel='', ylabel='targets',
                    xticklabels=[],
                    yticklabels=ytl)

        plot_matrix(ax=plt.subplot(gs_c2[row+1, col]),
                    data=Z_freq,
                    title=r'$\,Z^\mathrm{freq}$ ' + freq,
                    xlabel=xl, ylabel=yl,
                    xticklabels=xtl,
                    yticklabels=ytl,
                    xticklabelrotation=True)
    return axes


def plot_boxcharts(
    gs, data, populations, pop_labels, pop_colors, xlabel='', ylabel='',
        xticklabels=True, ylims=False):
    """
    Plots boxcharts of generic data.
    """
    ax = plt.subplot(gs)
    for loc in ['top', 'right']:
        ax.spines[loc].set_color('none')

    data_plot = []
    for X in populations:
        # remove potential NANs
        data_X = data[X][~np.isnan(data[X])]
        data_plot.append(data_X)

    # ignore all warnings, target in particular VisibleDeprecationWarning
    # (could be removed in some cases with np.array(data_plot, dtype=object))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        boxes = ax.boxplot(
            data_plot,
            labels=pop_labels[:-1],
            sym='', showmeans=True, patch_artist=True,
            meanprops={'mec': 'white',
                       'marker': '_',
                       'markersize':
                       matplotlib.rcParams['lines.markersize'] * 0.5},
            medianprops={'color': 'k'},
            whiskerprops={'color': 'k', 'linestyle': '-'})

        for i, box in enumerate(boxes['boxes']):
            box.set_color(pop_colors[i])

    plt.xticks(rotation=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not xticklabels:
        ax.set_xticklabels([])

    if ylims:
        ax.set_ylim(ylims[0], ylims[1])

    ax.yaxis.set_major_locator(MaxNLocator(3))
    return ax


def plot_barcharts(
    gs, data, populations, pop_labels, pop_colors, xlabel='', ylabel='',
        xticklabels=True, ylims=False):
    """
    Plots barcharts of generic data.
    """
    ax = plt.subplot(gs)
    for loc in ['top', 'right']:
        ax.spines[loc].set_color('none')

    if len(data) != len(populations):
        raise Exception
    xs = np.arange(len(data))
    ax.bar(x=xs,
           height=data,
           color=pop_colors[:-1])

    ax.set_xticks(xs)
    ax.set_xticklabels(pop_labels[:-1])
    plt.xticks(rotation=90)
    if not xticklabels:
        ax.set_xticklabels([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylims:
        ax.set_ylim(ylims[0], ylims=[1])

    ax.yaxis.set_major_locator(MaxNLocator(3))
    return ax


def plot_layer_panels(
    gs, plotfunc, populations, layer_labels, xlabel='', ylabel='', ylims=False,
        **kwargs):
    """
    Generic function to plot four vertically arranged panels, one for each
    layer, iterating over populations.
    """
    gs_c = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=gs)

    layer_count = 0
    for i, X in enumerate(populations):
        # select subplot
        if i > 0 and i % 2 == 0:
            layer_count += 1
        if i % 2 == 0:
            ax = plt.subplot(gs_c[layer_count])
            for loc in ['top', 'right']:
                ax.spines[loc].set_color('none')

        # specific plot
        plotfunc(ax, X, i, **kwargs)

        if ylims:
            ax.set_ylim(ylims[0], ylims[1])
        else:
            if i % 2 == 0:
                ymin, ymax = ax.get_ylim()
            if i % 2 == 1:
                ymin1, ymax1 = ax.get_ylim()

                if ax.get_yscale() == 'log':
                    ax.set_yticks([10.**x for x in np.arange(-15, 15)])
                    y0 = np.min([ymin, ymin1])
                else:
                    y0 = 0

                ax.set_ylim(np.min([ymin, ymin1, y0]),
                            np.max([ymax, ymax1]) * 1.1)

        if layer_count == len(layer_labels) - 1:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])

        if i == 0:
            ax.set_ylabel(ylabel)
            ax_label = ax
    return ax_label


def plot_population_panels(
        gs,
        plotfunc,
        populations,
        xlabel='',
        ylabel='',
        ylim_top=False,
        yticklabels=True,
        **kwargs):
    """
    Generic function to plot vertically arranged panels, one for each
    population.
    """
    num_pops = len(populations)

    gs_c = gridspec.GridSpecFromSubplotSpec(
        num_pops, 1, subplot_spec=gs)

    for i, X in enumerate(populations):
        # select subplot
        ax = plt.subplot(gs_c[i])
        for loc in ['top', 'right']:
            ax.spines[loc].set_color('none')

        # specific plot
        plotfunc(ax, X, i, **kwargs)

        if i == num_pops - 1:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])

        if i == 0:
            ax.set_ylabel(ylabel)
            ax_label = ax

        if ylim_top:
            ax.set_ylim(top=ylim_top)

        if not yticklabels:
            ax.set_yticklabels([])
    return ax_label


def plot_matrix(
        ax, data, title='', xlabel='', ylabel='',
        xticklabels=[], yticklabels=[], xticklabelrotation=False,
        cmap='coolwarm', vmin=-1, vmax=1):
    """
    Plots matrix data with imshow.
    """
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xticks((np.arange(len(data))))
    ax.set_yticks((np.arange(len(data[0]))))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    if xticklabelrotation:
        plt.xticks(rotation=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return


def plot_parameters_matrix(
        ax, data, pop_labels, title='',
        show_num='unique', num_format='{:.0f}', num_fontsize_scale=0.6,
        cmap='viridis', set_bad=[], cbar=True, vmin=None, vmax=None):
    """
    Plots matrix for network parameters.

    Parameters
    ----------
    show_num
        options are 'all', 'unique', False
    """
    tgt = np.shape(data)[0]  # numbers of rows
    src = np.shape(data)[1]  # number of columns
    block_data = np.zeros((tgt + 1, src + 1))

    # go through data:
    # 1. upper left    2. upper right
    # 3. lower left    4. lower right

    # extract data
    e_to_e = data[::2, ::2]
    i_to_e = data[::2, 1::2]
    e_to_i = data[1::2, ::2]
    i_to_i = data[1::2, 1::2]

    tgt_e = np.shape(e_to_e)[0]
    src_e = np.shape(e_to_e)[1]
    tgt_i = np.shape(i_to_i)[0]
    src_i = np.shape(i_to_i)[1]

    # upper left
    block_data[:tgt_e, :src_e] = e_to_e
    # upper right
    block_data[:tgt_e, src_e + 1:] = i_to_e
    # lower left
    block_data[tgt_e + 1:, :src_e] = e_to_i
    # lower right
    block_data[tgt_e + 1:, src_e + 1:] = i_to_i

    # set bad: separator and additional values
    block_data[tgt_e, :] = np.nan
    block_data[:, src_e] = np.nan
    for bad_val in set_bad:
        block_data[np.where(block_data == bad_val)] = np.nan

    # image
    block_data = np.ma.masked_invalid(block_data)
    cm = matplotlib.cm.get_cmap(cmap).copy()
    cm.set_bad('white')

    im = ax.imshow(block_data, cmap=cm, vmin=vmin,
                   vmax=vmax, aspect='auto')

    # annotate with numbers
    if show_num:
        _plot_parameters_show_numbers(
            ax, block_data, show_num, num_format, num_fontsize_scale)

    # annotate with panel titles
    for (loc, txt) in zip(
        [[(src_e - 1.) / 2., -1.], [src_e + 1 + (src_i - 1.) / 2., -1.],
            [(src_e - 1.) / 2., tgt_e], [src_e + 1 + (src_i - 1.) / 2., tgt_e]],
        [r'E$\rightarrow$E', r'I$\rightarrow$E',
            r'E$\rightarrow$I', r'I$\rightarrow$I']):
        ax.text(loc[0], loc[1] + 0.1, txt,
                ha='center', va='center')

    # replace frame
    [ax.spines[l].set_linewidth(0.) for l in [
        'top', 'bottom', 'left', 'right']]
    for (xy, width, height) in zip(
        [(-0.5, -0.5), (-0.5 + src_e + 1, -0.5),
            (-0.5, -0.5 + tgt_e + 1), (-0.5 + src_e + 1, -0.5 + tgt_e + 1)],
        [src_e, src_i, src_e, src_i],
            [tgt_e, tgt_e, tgt_i, tgt_i]):
        rec = Rectangle(xy=xy, width=width, height=height,
                        facecolor='none', edgecolor='k',
                        linewidth=matplotlib.rcParams['lines.linewidth'],
                        zorder=10)
        ax.add_patch(rec)

    # ticks and labels
    src_labels = np.zeros(src, dtype=object)
    src_labels[:src_e] = pop_labels[:src:2]
    src_labels[src_e:] = pop_labels[1:src:2]
    src_ticks = list(np.arange(src + 1))
    src_ticks.pop(src_e)
    ax.set_xticks(src_ticks)
    plt.xticks(rotation=90)
    ax.set_xticklabels(src_labels)
    ax.set_xlabel('sources')

    tgt_labels = np.zeros(tgt, dtype=object)
    tgt_labels[:tgt_e] = pop_labels[:tgt:2]
    tgt_labels[tgt_e:] = pop_labels[1:tgt:2]
    tgt_ticks = list(np.arange(tgt + 1))
    tgt_ticks.pop(tgt_e)
    ax.set_yticks(tgt_ticks)
    ax.set_yticklabels(tgt_labels)
    ax.set_ylabel('targets')

    if cbar:
        colorbar(ax, im, title, powerlimits=True)
    return


def plot_parameters_vector(
        ax, data, pop_labels, title='',
        show_num='unique', num_format='{:.0f}', num_fontsize_scale=0.6,
        cmap='viridis', set_bad=[], cbar=True, cbar_size='25%', vmin=None, vmax=None):
    """
    Plots vector for network parameters.

    Parameters
    ----------
    show_num
        options are 'all', 'unique', False
    """
    num = len(data)
    num_e = int(np.ceil(num / 2))
    num_i = num - num_e
    col_data = np.zeros(num + 1)

    # extract data
    col_data[:num_e] = data[::2]
    col_data[num_e + 1:] = data[1::2]

    # set bad: separator and additional values
    col_data[num_e] = np.nan

    for bad_val in set_bad:
        col_data[np.where(col_data == bad_val)] = np.nan

    # image
    col_data = col_data.reshape(-1, 1)  # column
    col_data = np.ma.masked_invalid(col_data)
    cm = matplotlib.cm.get_cmap(cmap).copy()
    cm.set_bad('white')

    im = ax.imshow(col_data, cmap=cm, vmin=vmin, vmax=vmax, aspect='auto')

    # annotate with numbers
    if show_num:
        _plot_parameters_show_numbers(
            ax, col_data, show_num, num_format, num_fontsize_scale)

    # replace frame
    [ax.spines[l].set_linewidth(0.) for l in [
        'top', 'bottom', 'left', 'right']]
    for (xy, width, height) in zip(
        [(-0.5, -0.5), (-0.5, 0.5 + num_e)],
        [1, 1],
            [num_e, num_i]):
        rec = Rectangle(xy=xy, width=width, height=height,
                        facecolor='none', edgecolor='k',
                        linewidth=matplotlib.rcParams['lines.linewidth'],
                        zorder=10)
        ax.add_patch(rec)

    # ticks and labels
    labels = np.zeros(num, dtype=object)
    labels[:num_e] = pop_labels[:num:2]
    labels[num_e:] = pop_labels[1:num:2]
    ticks = list(np.arange(num + 1))
    ticks.pop(num_e)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xticks([])

    if cbar:
        colorbar(ax, im, title, powerlimits=True, size=cbar_size)
    return


def _plot_parameters_show_numbers(
        ax, data, show_num, num_format, num_fontsize_scale):
    """
    Adds numbers to parameters plot.
    """
    if show_num == 'unique':
        values, flat_indices = np.unique(data, return_index=True)
        indices = np.unravel_index(flat_indices, np.shape(data))
    elif show_num == 'all':
        values = data.flatten()
        indices = np.indices(np.shape(data)).reshape(2, -1)

    for i, val in enumerate(values):
        if val != np.nan:
            txt = ax.text(
                indices[1][i],
                indices[0][i],
                num_format.format(val),
                ha='center',
                va='center',
                color='white',
                fontsize=matplotlib.rcParams['font.size'] *
                num_fontsize_scale)
            txt.set_path_effects(
                [PathEffects.withStroke(linewidth=1, foreground='k')])
    return


def plotfunc_distributions(
        ax, X, i, bins, data, pop_colors, MaxNLocatorNBins):
    """
    Inner plot function for distributions.
    """
    ax.hist(
        data[X],
        bins=bins,
        density=True,
        histtype='step',
        linewidth=matplotlib.rcParams['lines.linewidth'],
        color=pop_colors[i])

    ax.set_xlim(bins[0], bins[-1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=MaxNLocatorNBins))
    ax.set_yticks([])
    return


def plotfunc_PSDs(
        ax, X, i, data, pop_colors, psd_max_freq):
    """
    Inner plot function for power spectral densities.
    """
    # return if no data
    if isinstance(data[X], h5py._hl.dataset.Dataset) and data[X].size == 0:
        return

    freq = data[X]['frequencies_s-1']
    Pxx = data[X]['psds_s^-2_Hz-1']

    # skip frequency of 0 Hz in loglog plot
    freq = freq[1:]
    Pxx = Pxx[1:]
    ax.loglog(freq, Pxx,
              linewidth=matplotlib.rcParams['lines.linewidth'],
              color=pop_colors[i])

    ax.set_xticks([10**x for x in np.arange(1, 6)])
    ax.set_xlim(right=psd_max_freq)
    return


def plotfunc_CCs_distance(
        ax,
        X,
        i,
        data,
        key_ccs,
        pop_colors,
        max_num_pairs=10000,
        markersize_scale=0.4,
        nblocks=3):
    """
    Inner plot function for cross-correlation coefficients with distance.
    """
    # return if no data
    if isinstance(data[X], h5py._hl.dataset.Dataset) and data[X].size == 0:
        return

    distances = data[X]['distances_mm'][:max_num_pairs]
    ccs = data[X][key_ccs + 'ms'][:max_num_pairs]

    # loop for reducing zorder-bias
    blocksize = int(len(distances) / nblocks)
    for b in np.arange(nblocks):
        indices = np.arange(b * blocksize, (b + 1) * blocksize)
        zorder = 2 * b + i % 2  # alternating for populations

        ax.plot(
            distances[indices],
            ccs[indices],
            marker='$.$',
            markersize=matplotlib.rcParams['lines.markersize'] *
            markersize_scale,
            color=pop_colors[i],
            markeredgecolor='none',
            linestyle='',
            zorder=zorder,
            label=X.replace('L23', 'L2/3') if b == 0 else '_nolabel_',
            rasterized=False)
    return


def plotfunc_instantaneous_rates(
        ax,
        X,
        i,
        sptrains,
        num_neurons,
        pop_colors,
        time_step,
        time_interval):
    """
    Inner plot function for instantaneous firing rates.
    """
    data = load_h5_to_sparse_X(X, sptrains)

    # slice according to time interval
    time_indices = np.arange(
        time_interval[0] / time_step,
        time_interval[1] / time_step).astype(int)
    data = data[:, time_indices]

    # sum over neurons, divide by population size, and scale spike count to
    # instantaneous rate in spikes / s
    data = np.sum(data.toarray(), axis=0)
    data = data.astype(float) / num_neurons[i]
    data = data / (time_step * 1e-3)

    times = time_indices * time_step
    ax.fill_between(x=times, y1=0, y2=data,
                    color=pop_colors[i])

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(bottom=0)
    return


def plotfunc_theory_power_spectra(ax, X, i, data, pop_colors):
    """
    Inner plot function for power spectra calculated from theory.
    """
    frequencies = data[0]
    power = data[1]

    ax.plot(frequencies, power[:, i],
            linewidth=matplotlib.rcParams['lines.linewidth'],
            color=pop_colors[i])

    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    return


def adjust_lightness(color, amount=0.5):
    """
    Function from: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def colorbar(
        ax,
        im,
        label,
        axis='right',
        size='5%',
        pad=0.05,
        nbins=5,
        powerlimits=False,
        **kwargs):
    """
    Custom color bar.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(axis, size=size, pad=pad)
    cb = plt.colorbar(im, cax=cax, label=label, **kwargs)
    cb.locator = MaxNLocator(nbins=nbins)
    if powerlimits:
        cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()  # necessary for location
    return cb


def add_label(ax, label, offset=[0, 0],
              weight='bold', fontsize_scale=1.2):
    """
    Adds label to axis with given offset.

    Parameters
    ----------
    ax
        Axis to add label to.
    label
        Label should be a letter.
    offset
        x-,y-Offset.
    weight
        Weight of font.
    fontsize_scale
        Scaling factor for font size.
    """
    label_pos = [0. + offset[0], 1. + offset[1]]
    ax.text(label_pos[0], label_pos[1], label,
            ha='left', va='bottom',
            transform=ax.transAxes,
            weight=weight,
            fontsize=matplotlib.rcParams['font.size'] * fontsize_scale)
    return


def savefig(
        data_dir_circuit, extension, filename,
        eps_conv=False, eps_conv_via='.svg'):
    """
    Saves the current figure to format given in the plotting parameters.

    Note that inkscape etc. for conversion are not available on JURECA.

    Parameters
    ----------
    data_dir_circuit
        Path to data directory
    extension
        File extension.
    filename
        Name of the file.
    eps_conv
        If the format is .eps and eps_conv=True, the .eps file is converted
        to .pdf and back to .eps to properly compress rasterized parts of
        the figure.
        This is slow but gives a good result with small file size.
    eps_conv_via
        Options are '.svg' (using inkskape) and '.pdf' (using epstopdf and
        pdftops).
    """

    path_fn = os.path.join(data_dir_circuit, 'plots', filename)

    if extension == '.eps' and eps_conv:

        if eps_conv_via == '.svg':
            prior_ext = '.svg'
            plt.savefig(path_fn + prior_ext)
            cmd = ('inkscape ' + path_fn + '.svg ' +
                   '-E ' + path_fn + '.eps ' +
                   '--export-ignore-filters --export-ps-level=3' + '\n' +
                   'rm ' + path_fn + '.svg')
            os.system(cmd)

        elif eps_conv_via == '.pdf':
            prior_ext = '.eps'
            plt.savefig(path_fn + prior_ext)
            cmd = ('epstopdf ' + path_fn + '.eps' + '\n' +
                   'pdftops -eps ' + path_fn + '.pdf' + '\n' +
                   'rm ' + path_fn + '.pdf')
            os.system(cmd)

    else:
        plt.savefig(path_fn + extension)

    plt.close()
    return
