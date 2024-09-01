#!/usr/bin/env python
'''Plotting script for LFPs

'''
import sys
import h5py
import pandas as pd
import mesocircuit.mesocircuit_framework as mesoframe
import mesocircuit.plotting.plotting as plot
from mesocircuit.parameterization.base_plotting_params import rcParams
from mesocircuit.lfp.compute_mua import write_mua_file
import mesocircuit.lfp.plotting as lfpplt
from mesocircuit.lfp.lfp_parameters import get_parameters
from hybridLFPy import CachedTopoNetwork
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib
import os


##########################################################################
# Network parameters
##########################################################################

###############################################################################
# Instantiate a Mesocircuit object with parameters from the command line:
# the general data directory data_dir, the name of the experiment name_exp, and
# the ID of this parameterset ps_id.
# Previously evaluated parameters are loaded.

circuit = mesoframe.Mesocircuit(
    data_dir=sys.argv[-3], name_exp=sys.argv[-2], ps_id=sys.argv[-1],
    load_parameters=True)

sim_dict = circuit.sim_dict
net_dict = circuit.net_dict
ana_dict = circuit.ana_dict
plot_dict = circuit.plot_dict

# Set some matplotlib defaults
matplotlib.rcParams.update(rcParams)

##########################################################################
# LFP output directory
##########################################################################
path_lfp_data = os.path.join(circuit.data_dir_circuit, 'lfp')
path_fig_files = os.path.join(path_lfp_data, 'figures')

##########################################################################
# get ParameterSet object instance with all required parameters for LFPs
# etc.
##########################################################################
PS = get_parameters(path_lfp_data=path_lfp_data,
                    sim_dict=sim_dict,
                    net_dict=net_dict)


# plot signal simulation times for each cell type
df = pd.DataFrame(
    columns=[
        'y',
        'CachedNetwork',
        'Population',
        'run',
        'collect'])
for i, y in enumerate(PS.y):
    fname = os.path.join(path_lfp_data, f'simstats_{y}.dat')
    df_y = pd.read_csv(fname, delimiter=' ')
    df = pd.concat([
        df,
        pd.DataFrame(np.array([y] + df_y['time'].to_list(),
                              dtype='object'
                              ).reshape((1, -1)),
                     columns=['y', 'CachedNetwork',
                              'Population', 'run', 'collect'])
    ], ignore_index=True)

df.to_csv(os.path.join(path_lfp_data, 'simstats.csv'), index=False)

# sum
df['sum'] = np.round(
    df[['CachedNetwork', 'Population', 'run', 'collect']
       ].astype(float).sum(axis=1)).astype(int)
# per second of total simulation duration
df['per_s'] = np.round(df['sum'] /
                       (sim_dict['t_presim'] +
                        sim_dict['t_sim']) *
                       1E3).astype(int)

fig, ax = plt.subplots(1, 1,
                       figsize=(plot_dict['fig_width_1col'],
                                plot_dict['fig_width_1col']),
                       sharex=True)
df.plot(ax=ax)
ax.set_xticks(np.arange(len(PS.y)))
ax.set_xticklabels(PS.y, rotation='vertical')
ax.set_xlabel('cell type ($y$)')
ax.set_ylabel('time (s)')
ax.set_title('simulation time')


#########################################################################
# Create an object representation containing the spiking activity of the
# network simulation output that uses sqlite3.
##########################################################################
networkSim = CachedTopoNetwork(**PS.network_params)


##########################################################################
# Compute MUA signal and write to file
##########################################################################
if not os.path.isfile(os.path.join(path_lfp_data, PS.MUAFile)):
    write_mua_file(circuit,
                   PS, path_lfp_data, networkSim)


##########################################################################
# Create plots and animations
##########################################################################

# Figure 6 (old)
fig, ax = plt.subplots(1, 1,
                       figsize=(plot_dict['fig_width_2col'],
                                plot_dict['fig_width_2col'] * 0.5))
fig.subplots_adjust(left=0.13, right=1, bottom=0.0, top=1.)
lfpplt.morphology_table(ax, PS)
fig.savefig(os.path.join(path_fig_files, 'morphology_table.pdf'))

# Figure 6
fig, ax = plt.subplots(1, 1,
                       figsize=(plot_dict['fig_width_2col'],
                                plot_dict['fig_width_2col'] * 0.4))
fig.subplots_adjust(left=0.08, right=1, bottom=0.0, top=1.)
lfpplt.morphology_table(ax, PS, annotations=False)
fig.savefig(os.path.join(path_fig_files, 'figure_06.pdf'))


# Figure 7A
CONTACTPOS = ((600, 600), (600, 1000), (-1400, -1400))
fig, ax = plt.subplots(1, 1, figsize=(plot_dict['fig_width_1col'],
                                      plot_dict['fig_width_1col']),
                       sharex=True)
lfpplt.layout_illustration(ax, PS, net_dict, ana_dict, CONTACTPOS=CONTACTPOS)
fig.savefig(os.path.join(path_fig_files, 'contacts.pdf'))


# Figure 7B: plot LFP in each channel
fig, axes = plt.subplots(3, 1, figsize=(plot_dict['fig_width_1col'],
                                        plot_dict['fig_width_1col']))
T = [sim_dict['t_presim'], sim_dict['t_presim'] + 500]
fname = os.path.join(path_lfp_data, PS.electrodeFile)
lfpplt.plot_single_channel_lfp_data(axes[0], PS, fname,
                                    T=T, CONTACTPOS=CONTACTPOS,
                                    report_corrcoefs=True)
plt.setp(axes[0].get_xticklabels(), visible=False)

# Figure 7C: plot CSD in same channel
fname = os.path.join(path_lfp_data, PS.CSDFile)
lfpplt.plot_single_channel_csd_data(axes[1], PS, fname,
                                    T=T, CONTACTPOS=CONTACTPOS,
                                    report_corrcoefs=True)
plt.setp(axes[1].get_xticklabels(), visible=False)

# Figure 7D: plot MUA in same channel
fname = os.path.join(path_lfp_data, PS.MUAFile)
lfpplt.plot_single_channel_lfp_data(axes[2], PS, fname,
                                    T=T, CONTACTPOS=CONTACTPOS,
                                    title='MUA', ylabel=r'$s^{-1}$',
                                    report_corrcoefs=True)
axes[2].set_xlabel('time (ms)')

fig.savefig(os.path.join(path_fig_files, 'signal_timeseries_I.pdf'))


# Figure 7
fig = plt.figure(
    figsize=(
        plot_dict['fig_width_2col'],
        plot_dict['fig_width_1col']))
axes = []
gs = GridSpec(3, 4, wspace=0.7, hspace=0.3)

# Figure 7A
CONTACTPOS = ((600, 600), (600, 1000), (-1400, -1400))
ax = fig.add_subplot(gs[:, :2])
axes.append(ax)
lfpplt.layout_illustration(ax, PS, net_dict, ana_dict, CONTACTPOS=CONTACTPOS)

# Figure 7B: plot LFP in each channel
for i in range(3):
    axes.append(fig.add_subplot(gs[i, 2]))
T = [sim_dict['t_presim'], sim_dict['t_presim'] + 100]
fname = os.path.join(path_lfp_data, PS.electrodeFile)
lfpplt.plot_single_channel_lfp_data(axes[1], PS, fname,
                                    T=T, CONTACTPOS=CONTACTPOS)
plt.setp(axes[1].get_xticklabels(), visible=False)

# Figure 7C: plot CSD in same channel
fname = os.path.join(path_lfp_data, PS.CSDFile)
lfpplt.plot_single_channel_csd_data(axes[2], PS, fname,
                                    T=T, CONTACTPOS=CONTACTPOS)
plt.setp(axes[2].get_xticklabels(), visible=False)

# Figure 7D: plot MUA in same channel
fname = os.path.join(path_lfp_data, PS.MUAFile)
lfpplt.plot_single_channel_lfp_data(
    axes[3],
    PS,
    fname,
    T=T,
    CONTACTPOS=CONTACTPOS,
    title='MUA',
    ylabel=r'$\nu$ $(\mathrm{s}^{-1})$',
    subtract_mean=False)
axes[3].set_xlabel('time (ms)')

# Figure 7E-G: LFP/CSD/MUA power spectra
for i in range(3):
    axes.append(fig.add_subplot(gs[i, 3]))

fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]
ylabels = [r'$(\mathrm{mV}^2/\mathrm{Hz})$',
           r'$((\frac{\mathrm{nA}}{\mathrm{µm}^3})^2/\mathrm{Hz})$',
           r'$(\mathrm{s}^{-2}/\mathrm{Hz})$']
titles = [r'$PSD_\mathrm{LFP}$', r'$PSD_\mathrm{CSD}$', r'$PSD_\mathrm{MUA}$']
for i, (ax, fname, ylabel, title) in enumerate(zip(axes[4:], fnames,
                                                   ylabels, titles)):
    lfpplt.plot_spectrum(
        ax, fname, ylabel, title,
        psd_max_freq=plot_dict['psd_max_freq'],
        NFFT=ana_dict['psd_NFFT'],
        noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
        detrend='mean')
    if i < 2:
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel('frequency (Hz)')

for i, ax in enumerate(axes):
    plot.add_label(ax, 'ABCDEFG'[i])

fig.savefig(os.path.join(path_fig_files, 'figure_07.pdf'))


# Figure 8 LFP/CSD/MUA power spectra
fig, axes = plt.subplots(3, 1, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_1col']),
                         sharex=True)
fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]
ylabels = [r'$(\mathrm{mV}^2/\mathrm{Hz})$',
           r'$((\frac{\mathrm{nA}}{\mathrm{µm}^3})^2/\mathrm{Hz}})$',
           r'$(s^{-1})$']
titles = [r'$PSD_\mathrm{LFP}$', r'$PSD_\mathrm{CSD}$', r'$PSD_\mathrm{MUA}$']
for i, (ax, fname, ylabel, title) in enumerate(zip(axes, fnames,
                                                   ylabels, titles)):
    lfpplt.plot_spectrum(
        ax, fname, ylabel, title,
        psd_max_freq=plot_dict['psd_max_freq'],
        NFFT=ana_dict['psd_NFFT'],
        noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
        detrend='mean')
    if i < 2:
        plt.setp(ax.get_xticklabels(), visible=False)
fig.savefig(os.path.join(path_fig_files, 'signal_PSD.pdf'))


# Figure 8 LFP/CSD/MUA power spectra
fig, axes = plt.subplots(3, 1, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_1col']),
                         sharex=True)
fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]
ylabels = [r'$(\mathrm{mV}^2/\mathrm{Hz})$',
           r'$((\frac{\mathrm{nA}}{\mathrm{µm}^3})^2/\mathrm{Hz}})$',
           r'$(s^{-1})$']
titles = [r'$PSD_\mathrm{LFP}$', r'$PSD_\mathrm{CSD}$', r'$PSD_\mathrm{MUA}$']
for i, (ax, fname, ylabel, title) in enumerate(zip(axes, fnames,
                                                   ylabels, titles)):
    lfpplt.plot_spectrum(
        ax, fname, ylabel, title,
        psd_max_freq=plot_dict['psd_max_freq'],
        NFFT=ana_dict['psd_NFFT'],
        noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
        detrend='mean',
        plot_type='semilogy')
    if i < 2:
        plt.setp(ax.get_xticklabels(), visible=False)
fig.savefig(os.path.join(path_fig_files, 'signal_PSD_semilogy.pdf'))


# Figure 8 spike/LFP/CSD/MUA correlation vs. distance
fig, axes = plt.subplots(1, 3,
                         figsize=(plot_dict['fig_width_2col'],
                                  plot_dict['fig_width_1col']),
                         sharex=True)

fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]
for ax, data, fit_exp in zip(axes, fnames, [True, True, True]):
    lfpplt.plot_signal_correlation_or_covariance(
        ax=ax, PS=PS, data=data,
        extents=[net_dict['extent'] * 1E3] * 2,
        method=np.corrcoef,
        fit_exp=fit_exp)
fig.savefig(os.path.join(path_fig_files, 'correlations_w_distance.pdf'))


# Figure 8 LFP/CSD/MUA time series
fig, axes = plt.subplots(1, 3, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_2col']),
                         sharex=True, sharey=True)
fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]
units = ['mV', 'nA/µm3', '1/s']
titles = ['LFP', 'CSD', 'MUA']
for ax, fname, unit, title in zip(axes, fnames, units, titles):
    ax.set_prop_cycle('color', [plt.cm.gray(i)
                                for i in np.linspace(0, 200, 10).astype(int)])
    lfpplt.plot_signal_sum(
        ax, PS, fname, unit,
        T=[sim_dict['t_presim'], sim_dict['t_presim'] + 100],
        color=None)
    ax.set_title(title)

fig.savefig(os.path.join(path_fig_files, 'signal_timeseries_II.pdf'))


# Figure 8: new
fig, axes = plt.subplots(
    2, 2,
    figsize=(plot_dict['fig_width_2col'],
             plot_dict['fig_width_1col'] * 1.5),
    sharex=True)
fig.subplots_adjust(wspace=0.15, hspace=0.15)
axes = axes.flatten()

# Fig 8 A spike train correlations
# pairwise spike-train correlations with distance
nbins = 51
ylabel = 'corr. coeff.'
paneltitle = r'$CC_{\{t_s^i\}\{t_s^j\}}}$'
ax = axes[0]

# set up axes stealing space from main axes
divider = lfpplt.make_axes_locatable(ax)
axd = divider.append_axes("right", 0.25, pad=0.02, sharey=ax)
lfpplt.remove_axis_junk(ax)
lfpplt.remove_axis_junk(axd)
plt.setp(axd.get_yticklabels(), visible=False)

# file with precomputed pairwise correlations
try:
    # DUMB!!! - hardcode file paths to precomputed correlations
    fname = os.path.join(
        circuit.data_dir,
        'upscaled_CCs_only',
        'b09dcfc3953647c1a29c007fe8dde5fa',
        'processed_data',
        'all_CCs_distances.h5')
    assert os.path.isfile(fname)
except AssertionError:
    fname = os.path.join(
        circuit.data_dir_circuit,
        'processed_data',
        'all_CCs_distances.h5')
    print('NOTE: falling back to pairwise correlations ' +
          f'for figure_08.pdf in file {fname}')

with h5py.File(fname, 'r') as f:
    for i, (X, n_pairs) in enumerate(zip(['L23E', 'L23I'], [512, 512])):
        plot.plotfunc_CCs_distance(
            ax=ax, X=X, i=i, data=f,
            key_ccs='ccs_2.0ms',
            pop_colors=plot_dict['pop_colors'],
            max_num_pairs=n_pairs,
            markersize_scale=0.4,
            nblocks=3)
    ax.legend(loc='best', frameon=False, numpoints=3)

    # add entries with NaNs to mask
    for i, X in enumerate(['L23E', 'L23I']):
        c = f[X]['ccs_2.0ms'][()]
        mask = c != np.nan

        bins = np.linspace(c[mask].mean() - c[mask].std() * 1.5,
                           c[mask].mean() + c[mask].std() * 1.5,
                           nbins)
        axd.hist(c[mask], bins=bins, histtype='step', orientation='horizontal',
                 color=plot_dict['pop_colors'][i], clip_on=False, density=True)

# beautify
axd.set_xticks([])
ax.set_ylabel(ylabel, labelpad=0.1)
ax.set_title(paneltitle)

# Fig 8 B-D signal correlations with distance
fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]
paneltitles = [
    r'$CC_\mathrm{LFP}$',
    r'$CC_\mathrm{CSD}$',
    r'$CC_\mathrm{MUA}$']
for i, (ax, data, fit_exp, title) in enumerate(
        zip(axes[1:], fnames, [True, True, True], paneltitles)):
    ax_hist = lfpplt.plot_signal_correlation_or_covariance(
        ax=ax, PS=PS, data=data,
        extents=[net_dict['extent'] * 1E3] * 2,
        method=np.corrcoef,
        fit_exp=fit_exp)
    ax.set_title(title)
    if i == 0:
        ax_hist.set_xlabel('')
        ax.set_xlabel('')
    else:
        ax_hist.set_title('')

for i, ax in enumerate(axes):
    plot.add_label(ax, 'ABCD'[i])
    if i % 2 > 0:
        ax.set_ylabel('')
    if i < 2:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax.axis(ax.axis('tight'))

fig.savefig(os.path.join(path_fig_files, 'figure_08.pdf'))


# Figure 9: LFP/CSD/MUA coherences
fig, axes = plt.subplots(3, 1, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_1col']),
                         sharex=True, sharey=True)
fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]
titles = ['LFP', 'CSD', 'MUA']

for ax, fname, title in zip(axes, fnames, titles):
    lfpplt.plot_coherence_vs_frequency(
        ax, PS, fname, title,
        colors=plt.get_cmap('viridis', 5),
        NFFT=ana_dict['psd_NFFT'],
        noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
        method='mlab',
        tbin=ana_dict['binsize_time'],
        TRANSIENT=sim_dict['t_presim'])
fig.savefig(os.path.join(path_fig_files, 'signal_coherence.pdf'))


# Figure 9: LFP/CSD/MUA coherence with distance
fig, axes = plt.subplots(1, 3, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_1col']),
                         sharex=True, sharey=True)
fit_exps = [True, True, False]
for ax, fname, title, fit_exp in zip(axes, fnames, titles, fit_exps):
    lfpplt.plot_coherence_vs_distance(
        ax, PS, fname,
        max_inds=np.array([2, 6, 16, 26, 38]),
        NFFT=ana_dict['psd_NFFT'], noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
        method='mlab', tbin=ana_dict['binsize_time'],
        fit_exp=fit_exp,
        phase_coherence=False)
    if fit_exp:
        ax.legend(loc=1, fontsize=6, frameon=False,
                  bbox_to_anchor=(1.15, 1.05))
    ax.set_title(title)
fig.savefig(os.path.join(path_fig_files, 'signal_coherence_w_distance.pdf'))


# Figure 9: LFP/CSD/MUA coherence with frequency and distance
fig, axes = plt.subplots(1, 3, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_1col']),
                         sharex=True, sharey=True)
fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]
titles = ['LFP', 'CSD', 'MUA']

for ax, fname, title in zip(axes, fnames, titles):
    lfpplt.plot_coherence_vs_distance_vs_frequency(
        fig, ax, PS, fname,
        NFFT=ana_dict['psd_NFFT'],
        noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
        method='mlab',
        tbin=ana_dict['binsize_time'],
        TRANSIENT=sim_dict['t_presim'],
        title=title
    )
fig.savefig(
    os.path.join(
        path_fig_files,
        'signal_coherence_w_frequency_w_distance.pdf'))


# Fig 9:
fig, axes = plt.subplots(2, 3, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_1col'] * 1.5),
                         sharex='row', sharey='row')
fig.subplots_adjust(hspace=0.4, wspace=0.4)

fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile),
          os.path.join(path_lfp_data, PS.MUAFile)]

def _get_clims():
    # assess the 
    with h5py.File(fnames[0], 'r') as f:
        Fs = f['srate'][()]
        T0 = int(Fs * sim_dict['t_presim'] / 1000)  # t < T0 transient
        shape = f['data'].shape
        if len(shape) > 2:
            # flatten all but last axis
            data = f['data'][()].reshape((-1, shape[-1]))[:, T0:]
        else:
            data = f['data'][()][:, T0:]

    data_x = data
    data_y = data

    r, c, chfreqs, mask = lfpplt.get_data_coherence(
    data_x=data_x, data_y=data_y, srate=Fs,
    positions_x=PS.electrodeParams['x'],
    positions_y=PS.electrodeParams['y'],
    tbin=ana_dict['binsize_time'], NFFT=ana_dict['psd_NFFT'], noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
    method='mlab',
    phase_coherence=False)

    unique = np.unique(r[mask])
    means = []
    for d in unique:
        if d == 0:
            continue
        means += [c[mask][r[mask] == d].mean(axis=0)]
    means = np.array(means).T

    return [0, means[chfreqs >= 10, :].max()]

clim = _get_clims()

# panels A-C
titles = ['LFP', 'CSD', 'MUA']
for i, (ax, fname, title) in enumerate(zip(axes[0, ], fnames, titles)):
    lfpplt.plot_coherence_vs_frequency(
        ax, PS, fname, title,
        colors=plt.get_cmap('viridis', 5),
        NFFT=ana_dict['psd_NFFT'],
        noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
        method='mlab',
        tbin=ana_dict['binsize_time'],
        TRANSIENT=sim_dict['t_presim'],
        show_legend=True if i == 0 else False)
    if i > 0:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel('')
ax.axis('tight')
ax.set_ylim(clim)

# panels D-F
for i, (ax, fname, title) in enumerate(zip(axes[1, :], fnames, titles)):
    lfpplt.plot_coherence_vs_distance_vs_frequency(
        fig, ax, PS, fname,
        NFFT=ana_dict['psd_NFFT'],
        noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
        method='mlab',
        tbin=ana_dict['binsize_time'],
        TRANSIENT=sim_dict['t_presim'],
        title=title,
        show_cbar=(i == 2),
        clim=[clim[0], clim[1] * 0.5]
    )
    if i > 0:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel('')

for i, ax in enumerate(axes.flatten()):
    plot.add_label(ax, 'ABCDEF'[i])

fig.savefig(os.path.join(path_fig_files, 'figure_09.pdf'))
