from .plotting import Plotting
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pickle
import matplotlib
matplotlib.use('Agg')


def live_paper(data_dir, model, parameterview):
    """
    Creates a figure with raster, mesocircuit icon, and spatial snapshots.
    """
    # model path
    keys = list(parameterview[model]['paramsets'].keys())
    if len(keys) != 1:
        raise Exception
    else:
        hash = keys[0]
    model_path = os.path.join(data_dir, model, hash)

    # load data
    d = {}
    for all_datatype in ['all_sptrains', 'all_pos_sorting_arrays',
                         'all_inst_rates_bintime_binspace']:
        fn = os.path.join(
            model_path,
            'processed_data',
            all_datatype + '.h5')
        data = h5py.File(fn, 'r')
        d.update({all_datatype: data})

    # instantiate plotting
    dics = []
    for dic in ['sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
        with open(os.path.join(model_path, 'parameters', f'{dic}.pkl'),
                  'rb') as f:
            dics.append(pickle.load(f))
    sim_dict, net_dict, ana_dict, plot_dict = dics
    plot = Plotting(sim_dict, net_dict, ana_dict, plot_dict)

    #####

    print('Plotting live paper figure.')
    fig = plt.figure(figsize=(plot.plot_dict['fig_width_1col'],
                              plot.plot_dict['fig_width_1col']))
    gs = gridspec.GridSpec(5, 10)
    gs.update(left=0.12, right=0.96, bottom=0.08,
              top=1.02, hspace=0)

    # network sketch
    plot.plot_mesocircuit_icon(gs[0, -3:-1])

    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        5, 1, subplot_spec=gs[1:, :], hspace=0.35)

    fs = 8
    pad = 0.2
    plt.rc('axes', labelsize=fs)
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)

    # raster
    plot.plot_raster(
        gs_bottom[:3, 0],
        populations=plot.Y,
        all_sptrains=d['all_sptrains'],
        all_pos_sorting_arrays=d['all_pos_sorting_arrays'],
        time_step=plot.sim_dict['sim_resolution'],
        time_interval=plot.plot_dict['raster_time_interval_short'],
        sample_step=100,
        xlabel=False)
    plt.gca().tick_params(axis='both', which='major', pad=pad)
    plt.axvline(x=net_dict['th_start'], color='k')
    plt.title('TC', pad=pad, fontsize=fs)

    # spatial snapshots
    plot.plot_spatial_snapshots(
        gs_bottom[3:, 0],
        populations=plot.X,
        all_inst_rates_bintime_binspace=d['all_inst_rates_bintime_binspace'],
        binsize_time=plot.ana_dict['binsize_time'],
        orientation='horizontal',
        tickstep=4,
        cbar=False,
        cbar_left=0.28,
        cbar_width=0.01)
    plt.gca().tick_params(axis='both', which='major', pad=0.2)

    # TODO modify and use savefig
    plt.savefig(os.path.join(data_dir, 'live_paper.pdf'), dpi=600)
