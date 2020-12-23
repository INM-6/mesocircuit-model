#!/usr/bin/env python
'''Plotting script for LFPs

'''
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import sys
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hybridLFPy import CachedTopoNetwork
from core.parameterization.base_plotting_params import plot_dict
from core.lfp.lfp_parameters import get_parameters
from core.lfp.plotting import network_lfp_activity_animation, \
    morphology_table, layout_illustration, plot_single_channel_lfp_data, \
    plot_single_channel_csd_data, plot_spectrum, \
    plot_signal_correlation_or_covariance, plot_signal_sum


# Set some matplotlib defaults
matplotlib.rcParams.update(plot_dict['rcParams'])

##########################################################################
# Network parameters
##########################################################################
path_parameters = sys.argv[1]

dics = []
for dic in ['sim_dict', 'net_dict', 'ana_dict']:
    with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, ana_dict = dics

##########################################################################
# LFP output directory
##########################################################################
path_lfp_data = os.path.join(sim_dict['data_path'], 'lfp',
                             os.path.split(path_parameters)[-1])

##########################################################################
# get ParameterSet object instance with all required parameters for LFPs
# etc.
##########################################################################
PS = get_parameters(path_lfp_data=path_lfp_data,
                    sim_dict=sim_dict,
                    net_dict=net_dict)

#########################################################################
# Create an object representation containing the spiking activity of the
# network simulation output that uses sqlite3.
##########################################################################
networkSim = CachedTopoNetwork(**PS.network_params)

##########################################################################
# Create plots and animations
##########################################################################
'''# Figure 6
fig, ax = plt.subplots(1, 1,
                       figsize=(plot_dict['fig_width_2col'],
                                plot_dict['fig_width_2col'] * 0.5))
fig.subplots_adjust(left=0.13, right=1, bottom=0.0, top=1.)
morphology_table(ax, PS)


# Figure 7A
CONTACTPOS = (-200, 200)
fig, ax = plt.subplots(1, 1, figsize=(plot_dict['fig_width_1col'],
                                      plot_dict['fig_width_1col']),
                       sharex=True)
layout_illustration(ax, PS, net_dict, ana_dict, CONTACTPOS=CONTACTPOS)

# Figure 7B: plot LFP in one channel
fig, axes = plt.subplots(2, 1, figsize=(plot_dict['fig_width_1col'],
                                        plot_dict['fig_width_1col']))
T = [sim_dict['t_presim'], sim_dict['t_presim'] + 100]
fname = os.path.join(path_lfp_data, PS.electrodeFile)
plot_single_channel_lfp_data(axes[0], PS, net_dict, ana_dict, fname,
                             T=T, CONTACTPOS=CONTACTPOS)

# Figure 7C: plot CSD in same channel
fname = os.path.join(path_lfp_data, PS.CSDFile)
plot_single_channel_csd_data(axes[1], PS, net_dict, ana_dict, fname,
                             T=T, CONTACTPOS=CONTACTPOS)

# Figure 7D: plot MUA in same channel
# TODO!!
axes[1].set_xlabel('t (ms)')


# Figure 8 LFP/CSD/MUA power spectra
# TODO: add MUA
fig, axes = plt.subplots(1, 2, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_1col']),
                         sharex=True)

fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile)]
ylabels = [r'$(\mathrm{mV}^2/\mathrm{Hz})$',
           r'$((\frac{\mathrm{nA}}{\mathrm{µm}^3})^2/\mathrm{Hz}})$']
titles = [r'$PSD_\mathrm{LFP}$', r'$PSD_\mathrm{CSD}$']
for i, (ax, fname, ylabel, title) in enumerate(zip(axes, fnames,
                                                   ylabels, titles)):
    plot_spectrum(ax, fname, ylabel, title,
                  psd_max_freq=plot_dict['psd_max_freq'],
                  NFFT=ana_dict['psd_NFFT'],
                  noverlap=int(ana_dict['psd_NFFT'] * 3 // 4),
                  detrend='mean')

# Figure 8 spike/LFP/CSD/MUA correlation vs. distance
fig, axes = plt.subplots(1, 2,
                         figsize=(plot_dict['fig_width_2col'],
                                  plot_dict['fig_width_1col']),
                         sharex=True)

fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile)]
for ax, data, fit_exp in zip(axes, fnames, [True, False]):
    plot_signal_correlation_or_covariance(ax=ax, PS=PS, data=data,
                                          extent=net_dict['extent']*1E3,
                                          method=np.corrcoef,
                                          fit_exp=fit_exp)


# Figure 8 LFP/CSD/MUA time series
# TODO: MUA
fig, axes = plt.subplots(1, 2, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_2col']),
                         sharex=True, sharey=True)
fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile)]
units = ['mV', 'nA/µm3']
titles = ['LFP', 'CSD']
for ax, fname, unit, title in zip(axes, fnames, units, titles):
    ax.set_prop_cycle('color', [plt.cm.gray(i)
                                for i in np.linspace(0, 200, 10).astype(int)])
    plot_signal_sum(ax, PS, fname, unit, T=[500, 550])
    ax.set_title(title)
'''

import scipy.signal as ss


def get_data_coherence(analysis, PS,
                       data_x, data_y, srate,
                       tbin=5, NFFT=256, noverlap=192, method='mlab',
                       phase_coherence=False):
    # rate of downsampled signal
    srate_d = 1000 / tbin

    # downsample data by decimation
    q = srate / srate_d
    try:
        assert(q % int(q) == 0)
        q = int(q)
    except AssertionError as ae:
        raise ae('(tbin*1000) / srate must be even dividable')

    # downsampled data
    if q > 1:
        data_x_d = decimate(data_x, q=q, n=4, k=0.8, filterfun=ss.cheby1)
        data_y_d = decimate(data_y, q=q, n=4, k=0.8, filterfun=ss.cheby1)
    else:
        data_x_d = data_x
        data_y_d = data_y

    # z-transform data
    data_x_d = stats.ztransform(data_x_d)
    data_y_d = stats.ztransform(data_y_d)

    # position of contacts
    x = PS.electrodeParams['x'].astype(float) * 1E-3
    y = PS.electrodeParams['y'].astype(float) * 1E-3

    # distances between pairs of contacts
    r = np.zeros((x.size, y.size))

    for i in range(len(x)):
        r[i, ] = helperfun._calc_radial_dist_to_cell(
            x=x[i], y=y[i], Xpos=np.array(list(zip(x, y))),
            xextent=analysis.extent_length,
            yextent=analysis.extent_length,
            edge_wrap=analysis.pbc,
        )
    r = np.round(r, decimals=10)

    if phase_coherence:
        c = np.zeros((x.size, y.size, NFFT // 2 + 1)).astype('complex')
    else:
        c = np.zeros((x.size, y.size, NFFT // 2 + 1))
    if method == 'mlab':
        for i in range(x.size):
            Pxx, freqs = plt.mlab.psd(data_x_d[i, ],
                                      NFFT=NFFT, Fs=srate_d,
                                      noverlap=noverlap)
            for j in range(y.size):
                # fill in upper triangle
                if i <= j:
                    Pyy, _ = plt.mlab.psd(data_y_d[j, ],
                                          NFFT=NFFT, Fs=srate_d,
                                          noverlap=noverlap)
                    Pxy, _ = plt.mlab.csd(data_x_d[i, ], data_y_d[j, ],
                                          NFFT=NFFT, Fs=srate_d,
                                          noverlap=noverlap)
                    if phase_coherence:
                        c[i, j, ] = np.divide(Pxy, np.abs(Pxy))
                    else:
                        c[i, j, ] = np.sqrt(
                            np.divide(np.absolute(Pxy)**2, Pxx * Pyy))
    elif method == 'scipy':
        for i in range(x.size):
            freqs, Pxx = ss.welch(data_x_d[i, ],
                                  nperseg=NFFT, fs=srate_d,
                                  noverlap=noverlap)
            for j in range(y.size):
                # fill in upper triangle
                if i <= j:
                    _, Pyy = ss.welch(data_y_d[j, ],
                                      nperseg=NFFT, fs=srate_d,
                                      noverlap=noverlap)
                    _, Pxy = ss.csd(data_x_d[i, ], data_y_d[j, ],
                                    nperseg=NFFT, fs=srate_d,
                                    noverlap=noverlap)
                    if phase_coherence:
                        raise NotImplementedError
                    else:
                        c[i, j, ] = np.sqrt(
                            np.divide(np.absolute(Pxy)**2, Pxx * Pyy))

    # mask lower triangle of correlation/covariance matrices, mask
    # autocoherences/autocorrelatiosn
    mask = np.triu(np.ones((x.size, y.size)), k=1).astype(bool)

    return r, c, freqs, mask


# Figure 9: LFP/CSD/MUA coherences
# TODO: MUA
fig, axes = plt.subplots(1, 2, figsize=(plot_dict['fig_width_2col'],
                                        plot_dict['fig_width_2col']),
                         sharex=True, sharey=True)
fnames = [os.path.join(path_lfp_data, PS.electrodeFile),
          os.path.join(path_lfp_data, PS.CSDFile)]


# A-C: signal-signal coherence w. frequency
colors = plt.get_cmap('viridis', 5)
for i, (data, title) in enumerate(
        zip([LFP, np.array(CSD).T, MUA], ['LFP', 'CSD', 'MUA'])):
    ax = fig.add_subplot(gs[0, i])
    ax.set_title(
        r'$\langle\gamma_\mathrm{%s}\rangle (f)$' %
        (title + r'{\:}' + title))
    # dsp.annotate_subplot(ax, 3, 2, 'ABC'[i], 0.05, 16)
    dsp.add_label(ax, 'ABC'[i])

    data_x = data
    data_y = data
    for phase_coherence in [False, ]:
        r, c, chfreqs, mask = get_data_coherence(analysis=analysis, PS=PS,
                                                 data_x=data_x, data_y=data_y, srate=srate,
                                                 tbin=tbin, NFFT=NFFT, noverlap=noverlap, method=method,
                                                 phase_coherence=phase_coherence)
        unique = np.unique(r[mask])
        j = 0
        for d in unique[unique == np.round(unique, decimals=1)]:
            if d == 0:
                continue
            if phase_coherence:
                mean = np.abs(c[mask][r[mask] == d].mean(axis=0))
                finds = chfreqs <= 500.
                line = ax.plot(chfreqs[finds], mean[finds], ':', lw=1, alpha=1,
                               label='_nolabel_',
                               color=colors(j))
            else:
                mean = c[mask][r[mask] == d].mean(axis=0)
                finds = chfreqs <= 500.
                line = ax.plot(chfreqs[finds], mean[finds], '-', lw=1, alpha=1,
                               label='{} mm'.format(d),
                               color=colors(j))
            j += 1
    if i == 0:
        ax.set_ylabel('coherence', labelpad=0)
        # bbox_to_anchor=(1.1, 0.5), frameon=False)
        ax.legend(loc='best', fontsize=7, frameon=False, )
    ax.set_xlim(0, 500)
    dsp.remove_axis_junk(ax)
    ax.set_xlabel('$f$ (Hz)', labelpad=0)


'''
fig = network_lfp_activity_animation(
    PS, net_dict,
    networkSim, T=(100, 300),
    N_X=PS.N_X,
    save_anim=False)
# plt.close(fig)
'''

plt.show()
