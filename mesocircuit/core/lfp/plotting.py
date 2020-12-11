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
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
from matplotlib.collections import PolyCollection  # , LineCollection
# from matplotlib import mlab
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import h5py
# from scipy.optimize import curve_fit
# import scipy.signal as ss
# from scipy.signal import decimate
import numpy as np
# import sys
from ..parameterization.base_plotting_params import plot_dict
import LFPy
import matplotlib

# import matplotlib.style
# matplotlib.style.use('classic')

# Set some matplotlib defaults
matplotlib.rcParams.update(plot_dict['rcParams'])


pop_colors = plot_dict['pop_colors']


def network_lfp_activity_animation(PS, net_dict, networkSim, T=(
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

    dt = PS.dt_output
    tbins = np.arange(T[0], T[1] + dt, dt)

    if N_X is None:
        N_X = PS.N_X

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
        scat[X] = ax.scatter(
            spikes[X]['pos'][:, 0],
            spikes[X]['pos'][:, 1],
            s=np.random.rand(spikes[X]['size'].shape[0]) * 100,
            marker='^' if 'E' in X else '*',
            facecolors=networkSim.colors[j],
            edgecolors='none',
            label=X)

    # set legend
    ax.legend(loc=(0.65, -0.2), ncol=3, fontsize=10, frameon=False)

    # LFP data
    fname = os.path.join(PS.savefolder, 'PeriodicLFP_sum.h5')
    f = h5py.File(fname, 'r')
    data = f['data'][()]
    # subtract mean
    dataT = data.T - data.mean(axis=1)
    data = dataT.T
    # reshape
    data = data.reshape(
        (int(np.sqrt(PS.electrodeParams['x'].size)), -1, data.shape[-1]))
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
        for j, X in enumerate(PS.X):
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
                PS.savefolder,
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


def morphology_table(ax, PS):
    '''
    plot an illustration of the different morphology files used to represent
    each cell type

    Arguments
    ---------
    ax : matplotlib.figure.axes

    Returns
    -------

    '''
    # using colors assosiated with each main postsyn population
    colors = pop_colors

    layers = ['L1', 'L2/3', 'L4', 'L5', 'L6']

    # cell composition
    composition = PS.N_y.astype(float) / PS.N_y.sum() * 100

    morphotypes = [
        'p23',
        'i23',
        'i23',
        'p4',
        'ss4',
        'ss4',
        'i4',
        'i4',
        'p5v1',
        'p5v2',
        'i5',
        'i5',
        'p6',
        'p5v1',
        'i5',
        'i5',
    ]

    y_zip_list = list(zip(PS.y,
                          PS.m_y,
                          PS.depths,
                          PS.N_y,
                          composition,
                          morphotypes))

    xpos = 300
    xvec = [xpos]
    COUNTER = 0
    COUNTER_J = 0
    prevpop = None
    totnsegs = []

    for layerind, morpho, depth, size, relsize, mtype in y_zip_list:
        cell = LFPy.Cell(pt3d=False,
                         **PS.cellParams[layerind])

        cell.set_pos(xpos, 0, depth)

        totnsegs.append(cell.totnsegs)

        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z)))

        if COUNTER > 0 and prevpop != morpho.split('_')[0]:
            COUNTER_J += 1
        prevpop = morpho.split('_')[0]

        polycol = PolyCollection(zips,
                                 linewidths=0.5,
                                 edgecolors=colors[COUNTER_J],
                                 facecolors=colors[COUNTER_J],
                                 rasterized=False)
        ax.add_collection(polycol)

        xpos += 300

        xvec = np.r_[xvec, xpos]

        COUNTER += 1

    xvec = xvec[:-1]

    ax.hlines(PS.layerBoundaries[:, 0], 0, xpos - 100, linestyles='dotted')
    ax.hlines(PS.layerBoundaries[-1, -1], 0, xpos - 100, linestyles='dotted')
    ax.set_ylabel(r'depth ($\mu$m)', labelpad=0.1)
    ax.set_yticks(np.r_[PS.layerBoundaries[:, 0],
                        PS.layerBoundaries[-1, -1]])
    ax.set_xticks([])

    for i, z in enumerate(PS.layerBoundaries.mean(axis=1)):
        ax.text(0, z, layers[i], verticalalignment='center')

    for loc, spine in ax.spines.items():
        spine.set_color('none')  # don't draw spine
    ax.yaxis.set_ticks_position('left')

    ax.axis(ax.axis('equal'))

    # plot annotations
    i = 0
    j = 0
    xpos = 150
    # prevcolor = None
    prevpop = None
    for layerind, morpho, depth, size, relsize, mtype in y_zip_list:
        pop = morpho.split('_')[0]

        ax.text(xpos + 30, 300, '{:.1f}'.format(relsize),
                ha='left',
                clip_on=False)

        if i > 0 and prevpop != pop:
            ax.vlines(xpos, -1800, 900,
                      clip_on=False)
            j += 1

        if j > 7:  # HACK
            j = 7

        bigsize = PS.N_Y[j]

        ax.text(xpos + 30, 100, size, ha='left', clip_on=False)

        ax.text(xpos + 30, 200, '{:.1f}'.format(100 * float(size) / bigsize),
                ha='left')
        ax.text(xpos + 30, 400, '{}'.format(totnsegs[i]))
        ax.text(xpos + 30, 500, mtype,
                ha='left', clip_on=False)
        ax.text(xpos + 30, 600, layerind, ha='left', clip_on=False)

        if prevpop != pop:
            ax.text(xpos + 30, 700, bigsize, ha='left', clip_on=False)
            ax.text(xpos + 30, 800, pop.replace('23', '2/3'),
                    ha='left',
                    clip_on=False
                    )

        prevpop = pop
        xpos += 300

        i += 1

    ax.text(90, 800, r'Population $Y$:', ha='right', clip_on=False)
    ax.text(90, 700, r'Pop. size $N_Y$:', ha='right', clip_on=False)
    ax.text(90, 600, r'Cell type $y$:', ha='right', clip_on=False)
    ax.text(90, 500, r'Morphology $M_y$:', ha='right', clip_on=False)
    ax.text(90, 400, r'Segments $n_\mathrm{comp}$:', ha='right', clip_on=False)
    ax.text(90, 300, r'Occurrence $F_y$ (%):', ha='right', clip_on=False)
    ax.text(90, 200, r'Rel. Occurr. $F_{yY}$ (%):', ha='right', clip_on=False)
    ax.text(90, 100, r'Cell count $N_y$:', ha='right', clip_on=False)

    ax.axis(ax.axis('equal'))


def layout_illustration(ax, PS, net_dict, ana_dict, CONTACTPOS=(-200, 200)):
    '''
    Arguments
    ---------
    ax : matplotlib.axes._subplots.AxesSubplot
    PS : NeurotNeuroTools.parameters.ParameterSet
    net_dict: dict
        network settings
    ana_dict: dict
        analysis settings
    CONTACTPOS : tuple
        x and y coordinate of electrode contact point in PS.electrodeParams
    '''

    pos_bins = np.linspace(0, net_dict['extent'],
                           int(net_dict['extent'] / ana_dict['binsize_space']
                               ) + 1)
    pos_bins -= net_dict['extent'] / 2
    pos_bins *= 1E3  # mm -> µm

    # bin centers:
    xy = np.meshgrid(pos_bins[:-1], pos_bins[:-1])
    xy[0] += ana_dict['binsize_space'] / 2 * 1E3
    xy[1] += ana_dict['binsize_space'] / 2 * 1E3  # mm -> µm
    # get bin indices for slicing.
    CONTACT = (PS.electrodeParams['x'] == CONTACTPOS[0]
               ) & (PS.electrodeParams['y'] == CONTACTPOS[1])
    BINS = (xy[0] > CONTACTPOS[0] - 200) & (xy[0] < CONTACTPOS[0] + 200) & (
        xy[1] > CONTACTPOS[1] - 200) & (xy[1] < CONTACTPOS[1] + 200)
    BINS = BINS.flatten()

    ax.plot(PS.electrodeParams['x'], PS.electrodeParams['y'],
            'ko', markersize=5)
    ax.plot(PS.electrodeParams['x'][CONTACT], PS.electrodeParams['y'][CONTACT],
            'o', markersize=5, mfc='k', mec='k')
    for i, (x, y) in enumerate(zip(PS.electrodeParams['x'],
                                   PS.electrodeParams['y'])):
        ax.text(x, y - 250, '{}'.format(i + 1),
                color='k',
                horizontalalignment='center',
                verticalalignment='bottom',
                zorder=10)
    ax.plot(xy[0].flatten()[BINS], xy[1].flatten()[BINS], 's',
            markersize=5, zorder=-1, mfc='C2', mec='C2')
    ax.hlines(pos_bins, -2000, 2000, '0.8',
              clip_on=False, zorder=-2)
    ax.vlines(pos_bins, -2000, 2000, '0.8',
              clip_on=False, zorder=-2)
    ax.axis(ax.axis('equal'))
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([-1400, -1000], [-2100, -2100], 'k', lw=2, clip_on=False)
    ax.text(-1200, -2300, '0.4 mm', ha='center')

    ax.plot([1000, 2000], [-2100, -2100], 'k', lw=2, clip_on=False)
    ax.text(1500, -2300, '1 mm', ha='center')

    # axis cross
    ax.annotate("", xy=(-2100, -2100), xycoords='data',
                xytext=(-1600, -2100), textcoords='data',
                arrowprops=dict(arrowstyle="<|-",
                                connectionstyle="arc3,rad=0",
                                facecolor='black'),
                annotation_clip=False)
    ax.annotate("", xy=(-2100, -2100), xycoords='data',
                xytext=(-2100, -1600), textcoords='data',
                arrowprops=dict(arrowstyle="<|-",
                                connectionstyle="arc3,rad=0",
                                facecolor='black'),
                annotation_clip=False)
    ax.text(-1700, -2200, 'x', ha='center', va='center')
    ax.text(-2200, -1700, 'y', ha='center', va='center')
