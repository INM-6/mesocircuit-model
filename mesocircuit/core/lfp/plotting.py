#!/usr/bin/env python
'''prototype LFP simulation analysis'''
# from hybridLFPy.helpers import decimate
# import LFPy
# import dataset_plotting as dsp
# import dataset_analysis as dsa
# import parameterspace_control as psc
# from mesocircuit_LFP_parameters import get_LFP_parameters
# from meso_analysis import helpers, NetworkAnalysis, helperfun
# from matplotlib.gridspec import GridSpec
# from matplotlib.colors import LightSource
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.collections import PolyCollection, LineCollection
# from matplotlib import mlab
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import h5py
# from scipy.optimize import curve_fit
# import scipy.signal as ss
# from scipy.signal import decimate
import numpy as np
# import sys
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.style
matplotlib.style.use('classic')


def network_lfp_activity_animation(PSET, net_dict, networkSim, T=(
        500, 700), kernel=np.exp(-np.arange(10) / 2), N_X=None,
        save_anim=True):
    '''animation of network activity and LFP data'''
    fig, ax = plt.subplots(1, figsize=(9, 10))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.975)
    cbax = fig.add_axes([0.4, 0.1, 0.2, 0.02])

    ax.set_aspect('equal')

    ax.set_xlim((-net_dict['extent'] * 1E3 / 2,
                 net_dict['extent'] * 1E3 / 2))
    ax.set_ylim((-net_dict['extent'] * 1E3 / 2,
                 net_dict['extent'] * 1E3 / 2))
    ax.set_xlabel('x (um)', labelpad=0)
    ax.set_ylabel('y (um)', labelpad=0)
    ax.set_title('t=%.3i ms' % 100)

    dt = PSET.dt_output
    tbins = np.arange(T[0], T[1] + dt, dt)

    if N_X is None:
        N_X = PSET.N_X

    spikes = {}
    scat = {}
    for j, X in enumerate(networkSim.X):
        db = networkSim.dbs[X]
        gid = networkSim.nodes[X][:N_X[j]]
        gid_t = np.asarray(db.select_neurons_interval(gid, T), dtype=object)

        spikes[X] = np.zeros(gid_t.shape[0],
                             dtype=[('pos', float, 2),
                                    # dtype=[('pos', float,
                                    # networkSim.positions[X].shape),
                                    ('size', float, tbins.size - 1)])
        # set position arrays
        spikes[X]['pos'] = networkSim.positions[X][:N_X[j]]
        # set size arrays
        for i, t in enumerate(gid_t):
            spikes[X]['size'][i, :] = np.convolve(
                np.histogram(t, bins=tbins)[0] * 200, kernel, 'same')

        # scatter plot of positions, will not be shown in animation
        scat[X] = ax.scatter(spikes[X]['pos'][:,
                                              0],
                             spikes[X]['pos'][:,
                                              1],
                             s=np.random.rand(spikes[X]['size'].shape[0]) * 100,
                             marker='^' if 'E' in X else '*',
                             facecolors=networkSim.colors[j],
                             edgecolors='none',
                             label=X)

    # set legend
    ax.legend(loc=(0.65, -0.2), ncol=3, fontsize=10, frameon=False)

    # LFP data
    fname = os.path.join(PSET.savefolder, 'PeriodicLFP_sum.h5')
    f = h5py.File(fname, 'r')
    data = f['data'][()]
    # subtract mean
    dataT = data.T - data.mean(axis=1)
    data = dataT.T
    # reshape
    data = data.reshape(
        (int(np.sqrt(PSET.electrodeParams['x'].size)), -1, data.shape[-1]))
    # srate = f['srate'][()]

    # draw image plot on axes
    im = ax.pcolormesh(np.r_[0:4001:400] -
                       2000, np.r_[0:4001:400] -
                       2000, data[:, :, 0], vmin=-
                       data.std() *
                       4, vmax=data.std() *
                       4, zorder=-
                       1, cmap='RdBu')

    cbar = plt.colorbar(im, cax=cbax, orientation='horizontal')
    cbar.set_label('LFP (mV)', labelpad=0)
    tclbls = cbar.ax.get_xticklabels()
    plt.setp(tclbls, rotation=90, fontsize=10)

    def init():
        return

    def update(frame_number):
        '''update function for animation'''
        ind = frame_number % (tbins.size - 1)
        for j, X in enumerate(PSET.X):
            scat[X].set_sizes(spikes[X]['size'][:, ind])
        im.set_array(data[:, :, ind].flatten())
        ax.set_title('t=%.3i ms' % tbins[ind])

    ani = FuncAnimation(
        fig=fig,
        func=update,
        init_func=init,
        frames=tbins.size,
        interval=1)
    if save_anim:
        ani.save(
            os.path.join(
                PSET.savefolder,
                'hybridLFPyTopo.mp4'),
            fps=15,
            writer='ffmpeg',
            extra_args=[
                '-b:v',
                '5000k',
                '-r',
                '25',
                '-vcodec',
                'mpeg4'],
        )

    return fig
