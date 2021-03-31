"""PyNEST Mesocircuit: Figures
------------------------------

Definition of figures plotted with Plotting class in plotting.py.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import h5py
import matplotlib
matplotlib.use('Agg')


def parameters(plot):
    """
    Creates a figure with important parameter matrices and vectors.
    """
    orig_fontsize = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': orig_fontsize * 0.5})

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_2col'], 9))
    gs = gridspec.GridSpec(4, 3)
    gs.update(left=0.06, right=0.92, bottom=0.05, top=0.95,
              wspace=0.6, hspace=0.5)

    axes = []

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['full_num_synapses'],
        title='number of synapses',
        show_num=False,
        set_bad=[0])

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['full_indegrees'],
        title='in-degree',
        show_num='all',
        set_bad=[0])

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['beta'],
        title='beta (mm)',
        num_format='{:.3f}')

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['K_area_scaling'],
        title='in-degree scaling',
        num_format='{:.3f}')

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['delay_offset_matrix'],
        title='delay offset (ms)',
        num_format='{:.2f}')

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['prop_speed_matrix'],
        title='propagation speed (mm/ms)',
        num_format='{:.2f}')

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['delay_lin_eff_mean'],
        title='delay lin eff mean (ms)',
        num_format='{:.2f}')

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['delay_lin_eff_std'],
        title='delay lin eff std (ms)',
        num_format='{:.2f}')

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['full_weight_matrix_mean'],
        title='mean weight (pA)',
        num_format='{:.0f}')

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_matrix(
        axes[-1],
        data=plot.net_dict['p0_raw'],
        title='not full p0_raw',
        show_num='all', num_format='{:.2f}',
        set_bad=[0])

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_vector(
        axes[-1],
        data=plot.net_dict['full_num_neurons'],
        title='number of neurons',
        show_num=False)

    axes.append(plt.subplot(gs[len(axes)]))
    plot.plot_parameters_vector(
        axes[-1],
        data=plot.net_dict['full_ext_indegrees'],
        title='external in-degree',
        show_num='all')

    labels = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    for i, ax in enumerate(axes):
        plot.add_label(ax, labels[i])

    plot.savefig('parameters')

    plt.rcParams.update({'font.size': orig_fontsize})
    return


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

    if plot.net_dict['thalamic_input']:
        pops = plot.X
        num_neurons = plot.N_X
    else:
        pops = plot.Y
        num_neurons = plot.N_Y

    # automatically compute a samatplotlib. step for this figure
    if plot.plot_dict['raster_sample_step'] == 'auto':
        target_num_dots = 40000
        # assume an average firing rate of 4 Hz to estimate the number of
        # dots if all neurons were shown
        rate_estim = 4.
        full_num_dots_estim = \
            np.diff(plot.plot_dict['raster_time_interval']) * 1e-3 * \
            rate_estim * \
            np.sum(num_neurons)
        raster_sample_step = 1 + int(full_num_dots_estim / target_num_dots)
        print('  Automatically set raster_sample_step to ' +
              str(raster_sample_step) + '.')

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 5.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.98, bottom=0.1, left=0.17, right=0.92)
    ax = plot.plot_raster(
        gs[0, 0],
        pops,
        all_sptrains,
        all_pos_sorting_arrays,
        plot.sim_dict['sim_resolution'],
        plot.plot_dict['raster_time_interval'],
        raster_sample_step)

    plot.savefig('raster', eps_conv=True)
    return


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
        if isinstance(all_CCs_distances[X], h5py._hl.group.Group):
            all_CCs[X] = all_CCs_distances[X]['ccs']
        else:
            all_CCs[X] = np.array([])

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_2col'], 4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.98, bottom=0.15, top=0.95)
    axes = plot.plot_statistics_overview(
        gs[0], all_FRs, all_LVs, all_CCs, all_PSDs)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for i, label in enumerate(labels):
        plot.add_label(axes[i], label)

    plot.savefig('statistics_overview')
    return


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
    return


def spatial_snapshots(plot, all_inst_rates_bintime_binspace):
    """
    Creates a figure with consecutive snapshots.

    Parameters
    ----------
    plot
    all_inst_rates_bintime_binspace
    """
    print('Plotting spatial snapshots.')

    if plot.net_dict['thalamic_input']:
        pops = plot.X
    else:
        pops = plot.Y

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 4.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.17, right=0.97, top=0.99, bottom=0.2)
    ax = plot.plot_spatial_snapshots(
        gs[0, 0],
        pops,
        all_inst_rates_bintime_binspace,
        plot.ana_dict['binsize_time'],
        plot.ana_dict['binsize_space'])
    plot.savefig('spatial_snapshots')
    return


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
        print(
            'Not plotting cross-correlation functions with thalamic pulses ' +
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
    return


def instantaneous_firing_rates(plot, all_sptrains_bintime):
    """
    Creates a figure with histograms of instantaneous firing rates.
    """
    print('Plotting instantaneous firing rates.')

    if plot.net_dict['thalamic_input']:
        pops = plot.X
    else:
        pops = plot.Y

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'], 5.))
    gs = gridspec.GridSpec(1, 1)
    gs.update(top=0.98, bottom=0.1, left=0.17, right=0.92)
    ax = plot.plot_population_panels(
        gs[0, 0],
        plotfunc=plot.plotfunc_instantaneous_rates,
        populations=pops,
        xlabel='time (ms)',
        ylabel=r'$\nu (s^{-1})$',
        sptrains=all_sptrains_bintime,
        time_step=plot.ana_dict['binsize_time'],
        time_interval=plot.plot_dict['raster_time_interval'])

    plot.savefig('instantaneous_rates')
    return


def theory_overview(
        plot, working_point, frequencies, power,
        sensitvity_amplitude, sensitivity_frequency, sensitivity_popidx_freq):
    """
    """
    print('Plotting theory overview.')

    fig = plt.figure(figsize=(plot.plot_dict['fig_width_2col'], 4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right=0.98, bottom=0.15, top=0.93)
    axes = plot.plot_theory_overview(
        gs[0], working_point, frequencies, power,
        sensitvity_amplitude, sensitivity_frequency, sensitivity_popidx_freq)
    labels = ['A', 'B', 'C', 'D', 'E']
    for i, label in enumerate(labels):
        plot.add_label(axes[i], label)

    plot.savefig('theory_overview')
    return


def parameterspace_overviews(paramspace_key, data_dir):
    print('Plotting overviews for parameter space.')
    """
    TODO use auto data dir
    """
    
    import subprocess
    import pickle

    # ranges and hash map
    with open(os.path.join(data_dir, paramspace_key, 'parameter_space',
                           'parameters', 'ranges_hashmap.pkl'), 'rb') as f:
        hashmap_ranges = pickle.load(f)
    ranges = hashmap_ranges['ranges']
    hashmap = hashmap_ranges['hashmap']
    shape = np.shape(hashmap)
    rows = shape[0]
    if len(shape) == 1:
        cols = 1
        indices = np.arange(len(hashmap))
    elif len(shape) == 2:
        cols = shape[1]
        indices = np.zeros(shape, dtype=object)
        for r in np.arange(rows):
            for c in np.arange(cols):
                indices[r,c] = (r,c)
        indices = indices.flatten()

    # existing single figures (exclude parameter_space folder)
    sfigs = glob.glob(os.path.join(data_dir, paramspace_key,
                                   '[!parameter_space]*', 'plots', '*'))
    sfigs = np.unique([os.path.basename(p) for p in sfigs])

    for sf in sfigs:
        name, extension = sf.split('.')
        # TODO take first existing
        ext_file_name = os.path.join(data_dir, paramspace_key,
                                     hashmap[indices[0]], 'plots', sf)
        print(name, extension)
        if extension == 'pdf':
            pdfinfo = subprocess.check_output(['pdfinfo', ext_file_name]).decode('utf-8')
            for line in pdfinfo.split('\n'):
                if 'Page size' in line:
                    ps = line
            ps = ps.split(':')[1].split('pts')[0].split('x')
            sfig_size_pts = [float(s) for s in ps]
            sfig_size = [pts / 72 for pts in sfig_size_pts] # to inch
        else:
            raise Exception
        
        fig_size = [sfig_size[0] * cols, sfig_size[1] * rows]

        fig = plt.figure(figsize=fig_size)

        fname = os.path.join(data_dir, paramspace_key, 'parameter_space',
                             'plots', sf.split('.')[0])
        master_file_name = fname + '_master.pdf'
        fig.savefig(master_file_name)

        file = open('%s.tex' % fname , 'w')
        file.write(r"\documentclass{article}")
        file.write("\n")
        file.write(r"\usepackage{geometry}")
        file.write("\n")
        file.write(r"\geometry{paperwidth=%.3fin, paperheight=%.3fin, top=0pt, bottom=0pt, right=0pt, left=0pt}" % (fig_size[0],fig_size[1]))
        file.write("\n")
        file.write(r"\usepackage{tikz}")
        file.write("\n")
        file.write(r"\usepackage{graphicx}")
        file.write("\n")
        file.write(r"\pagestyle{empty}")
        file.write("\n")
        file.write(r"\begin{document}")
        file.write("\n")
        file.write(r"\noindent")
        file.write("\n")
        file.write(r"\resizebox{\paperwidth}{!}{")
        file.write("\n")
        file.write(r"  \begin{tikzpicture}")
        file.write("\n")
        file.write(r"    \node[inner sep=-1pt] (matplotlib_figure) at (0,0)")
        file.write("\n")
        file.write(r"    {\includegraphics{%s}};" % (master_file_name))
        file.write("\n")

        for ind in indices:
            if len(shape) == 1:
                xshift = -0.5*fig_size[0] + sfig_size[0] * (0.5 + ind)
                yshift = 0
            elif len(shape) == 2:
                xshift = -0.5*fig_size[0] + sfig_size[0] * (0.5 + ind[1])
                yshift = 0.5*fig_size[1] - sfig_size[1] * (0.5 + ind[0])

            pos_ext_figure =  (xshift, yshift)
            print(sfig_size, fig_size)
            print(ind, pos_ext_figure)


            ext_file_name = os.path.join(data_dir, paramspace_key,
                                         hashmap[ind], 'plots', sf)

            file.write(r"    \node[inner sep=-1pt,rectangle] (inkscape_sketch) at (%.4fin,%.4fin)" % (pos_ext_figure[0],pos_ext_figure[1]))
            file.write("\n")
            file.write(r"    {\includegraphics{%s}};" % (ext_file_name))
            file.write("\n")
  
        file.write(r"  \end{tikzpicture}")
        file.write("\n")
        file.write(r"}")
        file.write("\n")
        file.write(r"\end{document}")
        file.write("\n")

        file.close()

        # execute tex script
        os.system('pdflatex -output-directory=%s %s.tex' % (
            os.path.join(data_dir, paramspace_key, 'parameter_space','plots'),
            fname))

        #raise Exception

    return
