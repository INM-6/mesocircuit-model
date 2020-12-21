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
from hybridLFPy import CachedTopoNetwork
from core.parameterization.base_plotting_params import plot_dict
from core.lfp.lfp_parameters import get_parameters
from core.lfp.plotting import network_lfp_activity_animation, \
    morphology_table, layout_illustration, plot_single_channel_lfp_data, \
    plot_single_channel_csd_data


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
'''

# Figure 6A
CONTACTPOS=(-200, 200)
fig, ax = plt.subplots(1, 1, figsize=(plot_dict['fig_width_1col'],
                                      plot_dict['fig_width_1col']),
                       sharex=True)
layout_illustration(ax, PS, net_dict, ana_dict, CONTACTPOS=CONTACTPOS)

# Figure 6B : plot LFP in one channel
fig, axes = plt.subplots(3, 1, figsize=(plot_dict['fig_width_1col'],
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
axes[2].set_xlabel('t (ms)')


fig = network_lfp_activity_animation(
    PS, net_dict,
    networkSim, T=(100, 300),
    N_X=PS.N_X,
    save_anim=True)
# plt.close(fig)


plt.show()
