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
    plot_signal_correlation_or_covariance


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
'''
# Figure 6
fig, ax = plt.subplots(1, 1,
                       figsize=(plot_dict['fig_width_2col'],
                                plot_dict['fig_width_2col'] * 0.5))
fig.subplots_adjust(left=0.13, right=1, bottom=0.0, top=1.)
morphology_table(ax, PS)


# Figure 6A
CONTACTPOS = (-200, 200)
fig, ax = plt.subplots(1, 1, figsize=(plot_dict['fig_width_1col'],
                                      plot_dict['fig_width_1col']),
                       sharex=True)
layout_illustration(ax, PS, net_dict, ana_dict, CONTACTPOS=CONTACTPOS)

# Figure 6B : plot LFP in one channel
fig, axes = plt.subplots(2, 1, figsize=(plot_dict['fig_width_1col'],
                                        plot_dict['fig_width_1col']))
T = [sim_dict['t_presim'], sim_dict['t_presim'] + 100]
fname = os.path.join(path_lfp_data, PS.electrodeFile)
plot_single_channel_lfp_data(axes[0], PS, net_dict, ana_dict, fname,
                             T=T, CONTACTPOS=CONTACTPOS)

# Figure 6C: plot CSD in same channel
fname = os.path.join(path_lfp_data, PS.CSDFile)
plot_single_channel_csd_data(axes[1], PS, net_dict, ana_dict, fname,
                             T=T, CONTACTPOS=CONTACTPOS)

# Figure 6D: plot MUA in same channel
# TODO!!
axes[1].set_xlabel('t (ms)')


# Figure 7 LFP/CSD/MUA power spectra
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
                  detrend='mean')'''

# Figure 7 spike/LFP/CSD/MUA correlation vs. distance
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




'''
fig = network_lfp_activity_animation(
    PS, net_dict,
    networkSim, T=(100, 300),
    N_X=PS.N_X,
    save_anim=False)
# plt.close(fig)
'''

plt.show()
