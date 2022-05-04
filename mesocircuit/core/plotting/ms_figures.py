from .plotting import Plotting
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pickle
import matplotlib
matplotlib.use('Agg')


def reference_vs_upscaled(data_dir, ref_model, ups_model, parameterview):

    d = {}
    for m, model in enumerate([ref_model, ups_model]):

        # model path
        prefix = 'ref' if m == 0 else 'ups'
        keys = list(parameterview[model]['paramsets'].keys())
        if len(keys) != 1:
            raise Exception
        else:
            hash = keys[0]
        model_path = os.path.join(data_dir, model, hash)

        # load data
        for all_datatype in ['all_sptrains', 'all_pos_sorting_arrays',
                             'all_FRs', 'all_LVs', 'all_CCs_distances',
                             'all_PSDs']:
            fn = os.path.join(
                model_path,
                'processed_data',
                all_datatype + '.h5')
            data = h5py.File(fn, 'r')
            d.update({prefix + '_' + all_datatype: data})

        # instantiate plotting with first model
        if m == 0:
            dics = []
            for dic in ['sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
                with open(os.path.join(model_path, 'parameters', f'{dic}.pkl'),
                          'rb') as f:
                    dics.append(pickle.load(f))
            sim_dict, net_dict, ana_dict, plot_dict = dics
            plot = Plotting(sim_dict, net_dict, ana_dict, plot_dict)

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
