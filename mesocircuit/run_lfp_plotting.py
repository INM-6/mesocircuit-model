#!/usr/bin/env python
'''Plotting script for LFPs

'''
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from hybridLFPy import CachedTopoNetwork
from core.parameterization.base_plotting_params import plot_dict
from core.lfp.lfp_parameters import get_parameters
import core.lfp.plotting as lfpplt
from core.lfp.compute_mua import write_mua_file
import pandas as pd


# Set some matplotlib defaults
matplotlib.rcParams.update(plot_dict['rcParams'])

##########################################################################
# Network parameters
##########################################################################
# path_parameters = sys.argv[1]
path_parameters = 'parameters'

dics = []
for dic in ['sim_dict', 'net_dict', 'ana_dict']:
    with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, ana_dict = dics

##########################################################################
# LFP output directory
##########################################################################
# path_lfp_data = os.path.join(os.path.split(path_parameters)[0], 'lfp')
path_lfp_data = 'lfp'
path_fig_files = os.path.join('lfp', 'figures')

##########################################################################
# get ParameterSet object instance with all required parameters for LFPs
# etc.
##########################################################################
PS = get_parameters(path_lfp_data=path_lfp_data,
                    sim_dict=sim_dict,
                    net_dict=net_dict)


# plot signal simulation times for each cell type
df = pd.DataFrame(columns=['y', 'CachedNetwork', 'Population', 'run', 'collect'])
for i, y in enumerate(PS.y):
    fname = os.path.join(path_lfp_data, f'simstats_{y}.dat')
    df_y = pd.read_csv(fname, delimiter=' ')
    df = pd.concat([
        df,
        pd.DataFrame(np.array([y] + df_y['time'].to_list(), dtype='object').reshape((1, -1)), 
                     columns=['y', 'CachedNetwork', 'Population', 'run', 'collect'])
    ], ignore_index=True)

df.to_csv('simstats.csv', index=False)


df = pd.read_csv('simstats.csv')
# sum
df['sum'] =  np.round(df[['CachedNetwork', 'Population', 'run', 'collect']].sum(axis=1)).astype(int)
# per second of total simulation duration
df['per_s'] = np.round(df['sum'] / (sim_dict['t_presim'] + sim_dict['t_sim']) * 1E3).astype(int)

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

# print(df)
print(df[['y', 'per_s']])
print(df[['per_s']].to_numpy().flatten().tolist())

# plt.show()
# raise Exception

#########################################################################
# Create an object representation containing the spiking activity of the
# network simulation output that uses sqlite3.
##########################################################################
networkSim = CachedTopoNetwork(**PS.network_params)


##########################################################################
# Compute MUA signal and write to file
##########################################################################
if not os.path.isfile(os.path.join(path_lfp_data, PS.MUAFile)):
    write_mua_file(sim_dict, net_dict, ana_dict,
                   PS, path_lfp_data, networkSim)


##########################################################################
# Create plots and animations
##########################################################################

# Figure 6
fig, ax = plt.subplots(1, 1,
                       figsize=(plot_dict['fig_width_2col'],
                                plot_dict['fig_width_2col'] * 0.5))
fig.subplots_adjust(left=0.13, right=1, bottom=0.0, top=1.)
lfpplt.morphology_table(ax, PS)
fig.savefig(os.path.join(path_fig_files, 'morphology_table.pdf'))


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
lfpplt.plot_single_channel_lfp_data(axes[0], PS, net_dict, ana_dict, fname,
                                    T=T, CONTACTPOS=CONTACTPOS)
plt.setp(axes[0].get_xticklabels(), visible=False)

# Figure 7C: plot CSD in same channel
fname = os.path.join(path_lfp_data, PS.CSDFile)
lfpplt.plot_single_channel_csd_data(axes[1], PS, net_dict, ana_dict, fname,
                                    T=T, CONTACTPOS=CONTACTPOS)
plt.setp(axes[1].get_xticklabels(), visible=False)

# Figure 7D: plot MUA in same channel
fname = os.path.join(path_lfp_data, PS.MUAFile)
lfpplt.plot_single_channel_lfp_data(axes[2], PS, net_dict, ana_dict, fname,
                                    T=T, CONTACTPOS=CONTACTPOS,
                                    title='MUA', ylabel=r'$s^{-1}$')
axes[2].set_xlabel('t (ms)')

fig.savefig(os.path.join(path_fig_files, 'signal_timeseries_I.pdf'))


## Figure 7
fig = plt.figure(figsize=(plot_dict['fig_width_2col'], plot_dict['fig_width_1col']))
axes = []
gs = GridSpec(3, 2, wspace=0.4, hspace=0.3)

# Figure 7A
CONTACTPOS = ((600, 600), (600, 1000), (-1400, -1400))
ax = fig.add_subplot(gs[:, 0])
axes.append(ax)
lfpplt.layout_illustration(ax, PS, net_dict, ana_dict, CONTACTPOS=CONTACTPOS)

# Figure 7B: plot LFP in each channel
for i in range(3):
    axes.append(fig.add_subplot(gs[i, 1]))
T = [sim_dict['t_presim'], sim_dict['t_presim'] + 100]
fname = os.path.join(path_lfp_data, PS.electrodeFile)
lfpplt.plot_single_channel_lfp_data(axes[1], PS, net_dict, ana_dict, fname,
                                    T=T, CONTACTPOS=CONTACTPOS)
plt.setp(axes[1].get_xticklabels(), visible=False)

# Figure 7C: plot CSD in same channel
fname = os.path.join(path_lfp_data, PS.CSDFile)
lfpplt.plot_single_channel_csd_data(axes[2], PS, net_dict, ana_dict, fname,
                                    T=T, CONTACTPOS=CONTACTPOS)
plt.setp(axes[2].get_xticklabels(), visible=False)

# Figure 7D: plot MUA in same channel
fname = os.path.join(path_lfp_data, PS.MUAFile)
lfpplt.plot_single_channel_lfp_data(axes[3], PS, net_dict, ana_dict, fname,
                                    T=T, CONTACTPOS=CONTACTPOS,
                                    title='MUA', ylabel=r'$\nu$ $(\mathrm{s}^{-1})$',
                                    subtract_mean=False)
axes[3].set_xlabel('time (ms)')

for i, ax in enumerate(axes):
    lfpplt.add_label(ax, 'ABCD'[i])

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
    lfpplt.plot_signal_sum(ax, PS, fname, unit, 
                           T=[sim_dict['t_presim'], sim_dict['t_presim'] + 100])
    ax.set_title(title)

fig.savefig(os.path.join(path_fig_files, 'signal_timeseries_II.pdf'))



# Figure 9: LFP/CSD/MUA coherences
fig, axes = plt.subplots(1, 3, figsize=(plot_dict['fig_width_2col'],
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
        NFFT=256,
        noverlap=196,
        method='mlab',
        tbin=0.5,
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
        NFFT=256, noverlap=196,
        method='mlab', tbin=0.5,
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
        NFFT=256,
        noverlap=196,
        method='mlab',
        tbin=0.5,
        TRANSIENT=sim_dict['t_presim']
    )
fig.savefig(os.path.join(path_fig_files, 'signal_coherence_w_frequency_w_distance.pdf'))



'''fig = lfpplt.network_lfp_activity_animation(
    PS, net_dict,
    networkSim, T=(100, 300),
    N_X=PS.N_X,
    save_anim=True)
# plt.close(fig)'''


plt.show()
