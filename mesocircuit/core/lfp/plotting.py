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
from meso_analysis import helperfun
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import scipy.signal as ss



# import matplotlib.style
# matplotlib.style.use('classic')

# Set some matplotlib defaults
matplotlib.rcParams.update(plot_dict['rcParams'])


pop_colors = plot_dict['pop_colors']


def remove_axis_junk(ax, which=['right', 'top']):
    '''remove axis lines in list `which` from axes `ax`

    Parameters
    ----------
    ax :
    which: list of strings in ['right', 'top', 'bottom', 'left']
    '''
    for loc, spine in ax.spines.items():
        if loc in which:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

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
    ax: matplotlib.figure.axes

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
    ax: matplotlib.axes._subplots.AxesSubplot
    PS: NeurotNeuroTools.parameters.ParameterSet
    net_dict: dict
        network settings
    ana_dict: dict
        analysis settings
    CONTACTPOS: tuple
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


def plot_single_channel_lfp_data(ax, PS, net_dict, ana_dict, fname,
                                 title='LFP', ylabel=r'$\Phi$ (mV)',
                                 T=[500, 550], CONTACTPOS=(-200, 200)):
    '''
    Arguments
    ---------
    ax: matplotlib.axes._subplots.AxesSubplot
    PS: NeurotNeuroTools.parameters.ParameterSet
    net_dict: dict
        network settings
    ana_dict: dict
        analysis settings
    fname: str
        path to .h5 file
    title: str
    ylabel: str
    CONTACTPOS: tuple
        x and y coordinate of electrode contact point in PS.electrodeParams
    '''
    CONTACT = (PS.electrodeParams['x'] == CONTACTPOS[0]
               ) & (PS.electrodeParams['y'] == CONTACTPOS[1])
    f = h5py.File(fname, 'r')
    srate = f['srate'][()]
    data = f['data'][()][CONTACT, ].flatten()
    f.close()
    tinds = np.arange(T[0] * srate / 1000, T[1] * srate / 1000 + 1).astype(int)
    tvec = tinds.astype(float) / srate * 1000
    ax.plot(tvec, data[tinds] - data[tinds].mean(), 'k')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([])
    remove_axis_junk(ax)


def plot_single_channel_csd_data(
        ax, PS, net_dict, ana_dict, fname,
        title='CSD',
        ylabel=r'C ($\frac{\mathrm{nA}}{\mathrm{µm}^3}$)',
        T=[500, 550],
        CONTACTPOS=(-200, 200)):
    # CSD bin edges
    X, Y, Z = np.meshgrid(PS.CSDParams['x'],
                          PS.CSDParams['y'],
                          PS.CSDParams['z'])

    # midpoints (XY-plane)
    h = np.diff(PS.CSDParams['x'])[0]
    Xmid = X[:-1, :-1, 0] + h / 2
    Ymid = Y[:-1, :-1, 0] + h / 2

    # find CSD bin matching contact location
    CONTACT = (Xmid == CONTACTPOS[0]) & (Ymid == CONTACTPOS[1])

    f = h5py.File(fname, 'r')
    srate = f['srate'][()]
    data = f['data'][()][CONTACT, ].flatten()
    f.close()
    tinds = np.arange(T[0] * srate / 1000, T[1] * srate / 1000 + 1).astype(int)
    tvec = tinds.astype(float) / srate * 1000
    ax.plot(tvec, data[tinds] - data[tinds].mean(), 'k')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([])
    remove_axis_junk(ax)


def plot_spectrum(ax, fname,
                  ylabel=r'$(\mathrm{mV}^2/\mathrm{Hz})$',
                  title=r'$PSD_\mathrm{LFP}$',
                  psd_max_freq=500,
                  TRANSIENT=500,
                  **kwargs):
    """
    Plot average spectrum of multichannel data +- one standard deviation

    Parameters
    ----------
    ax: AxesSubplot
    fname: path
        Path to signal
    ylabel: str
        y-axis label
    title: str
        axis title
    psd_max_freq: float
        max frequency in plot
    TRANSIENT: float
        transient period removed from analysis (ms)
    **kwargs
        parameters to plt.mlab.psd
    """
    with h5py.File(fname, 'r') as f:
        Fs = f['srate'][()]
        T0 = int(Fs * TRANSIENT / 1000)  # t < T0 transient
        shape = f['data'].shape
        if len(shape) > 2:
            # flatten all but last axis
            data = f['data'][()].reshape((-1, shape[-1]))[:, T0:]
        else:
            data = f['data'][()][:, T0:]
    spectra = []
    for x in data:
        Pxx, freqs = plt.mlab.psd(x, Fs=Fs, **kwargs)
        spectra += [Pxx]
    spectra = np.c_[spectra]

    # discard high frequencies
    spectra = spectra[:, freqs <= psd_max_freq]
    freqs = freqs[freqs <= psd_max_freq]

    ax.loglog(freqs, spectra.mean(axis=0), 'k-', zorder=0)
    ax.fill_between(freqs,
                    spectra.mean(axis=0) - spectra.std(axis=0),
                    spectra.mean(axis=0) + spectra.std(axis=0),
                    color='gray', zorder=-1)
    ax.set_ylabel(ylabel)
    ax.grid('on')
    ax.set_title(title)
    ax.axis(ax.axis('tight'))
    remove_axis_junk(ax)


def plot_signal_correlation_or_covariance(
        ax,
        PS,
        data=None,
        srate=None,
        edge_wrap=True,
        extent=4000,
        TRANSIENT=500,
        method=np.cov,
        tbin=5,
        nbins=51,
        fit_exp=True):
    '''
    Compute and plot covariance and correlation coefficients between pairs
    of LFP channels as function of distance

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot
    PS: NeuroTools.parameters.ParameterSet
        LFP prediction parameters
    data: ndarray, str, None
        data to be analysed and plotted
    srate: float
        data sampling rate (Hz)
    TRANSIENT: float
        duration of transient period (ms)
    method: numpy.cov or numpy.corrcoef
    tbin: 5
        temporal binsize of downsampled LFP when computing correlations (ms)
    nbins: 51
        number of bins between C.min() and C.max() where C is the correlation
        or covariance matrix between pairs of channels
    fit_exp: bool
        Fit exponential curve to mean data points at each unique distance

    Returns
    -------
    axd: matplotlib.axes._axes.Axes
        axes object of histogram
    '''
    # rate of downsampled signal
    srate_d = 1000 / tbin

    if isinstance(data, str) and os.path.isfile(data):
        with h5py.File(data, 'r') as f:
            shape = f['data'].shape
            srate = f['srate'][()]
            T0 = int(srate * TRANSIENT / 1000)  # t < T0 transient
            if len(shape) > 2:
                # flatten all but last axis
                DATA = f['data'][()].reshape((-1, shape[-1]))[:, T0:]
            else:
                DATA = f['data'][()][:, T0:]
    elif isinstance(data, np.ndarray):
        DATA = data
        try:
            assert(srate > 0.)
        except AssertionError:
            raise AssertionError('srate must be a float > 0.')
    else:
        raise Exception('data not recognized')

    # downsample data by decimation
    q = srate / srate_d
    try:
        assert(q % int(q) == 0)
        q = int(q)
    except AssertionError as ae:
        raise ae('(tbin*1000) / srate must be even dividable')

    # downsampled DATA
    DATA_d = ss.decimate(DATA, q=q)

    # position of contacts
    x = PS.electrodeParams['x'].astype(float)
    y = PS.electrodeParams['y'].astype(float)

    # distances between pairs of contacts
    r = np.zeros((x.size, x.size))

    for i in range(len(x)):
        r[i, ] = helperfun._calc_radial_dist_to_cell(
            x=x[i], y=y[i], Xpos=np.array(list(zip(x, y))),
            xextent=extent,
            yextent=extent,
            edge_wrap=edge_wrap
        )

    # # correlation and covariance matrices
    c = method(DATA_d)

    # mask lower triangle of correlation/covariance matrices
    mask = np.triu(np.ones(c.shape), k=1).astype(bool)

    if method == np.cov:
        paneltitle = 'pairwise covariance'
        ylabel = 'covariance'
    else:
        paneltitle = 'pairwise correlation'
        ylabel = 'corr. coeff.'

    # add entries with NaNs to mask
    mask[c == np.nan] = False

    # highlight the average value at the different unique distances
    unique = np.unique(r[mask])
    mean = []
    std = []
    for v in unique:
        mean += [np.nanmean(c[mask][r[mask] == v])]
        std += [np.nanstd(c[mask][r[mask] == v])]

    mean = np.array(mean)
    std = np.array(std)

    ax.errorbar(unique / 1000., mean, yerr=std, xerr=None, fmt='ko-',
                label=ylabel)

    # set up axes stealing space from main axes
    divider = make_axes_locatable(ax)
    axd = divider.append_axes("right", 0.25, pad=0.02)

    bins = np.linspace(np.nanmin(c), np.nanmax(c), nbins)
    axd.hist(c[mask], bins=bins, histtype='step', orientation='horizontal',
             color='k', clip_on=False)

    # beautify
    ax.set_ylim(bins[0], bins[-1])
    axd.set_ylim(bins[0], bins[-1])
    axd.set_yticklabels([])
    axd.set_xticks([0, axd.axis()[1]])
    axd.set_title('distribution')

    ax.set_ylabel(ylabel, labelpad=0.1)
    ax.set_xlabel(r'$r$ (mm)', labelpad=0.1)
    axd.set_xlabel('count (-)', labelpad=0.1)
    ax.set_title(paneltitle)

    remove_axis_junk(ax)
    remove_axis_junk(axd)

    # fit exponential to values with distance
    if fit_exp:
        def func(x, a, b, c):
            '''cost function'''
            return a * np.exp(-x / b) + c

        # initial guess
        p0 = (.1, 100, 0.1)
        bounds = ([0, 0, 0], [1, 2000, 1])

        popt, pcov = curve_fit(func, r[mask], c[mask], p0=p0, bounds=bounds)

        # coeff of determination:
        residuals = c[mask] - func(r[mask], popt[0], popt[1], popt[2])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((c[mask] - np.mean(c[mask]))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # plot
        rvec = np.linspace(r[mask].min(), r[mask].max(), 101)
        ax.plot(rvec / 1000., func(rvec, *popt), 'r-', lw=1.5,
                label=r'$\beta=({0: .2g},{1: .2g},{2: .2g})$'.format(
                      popt[0], popt[1] / 1000, popt[2]
        ) + '\n' + r'$R^2={0: .2g}$'.format(r_squared)
        )

    # ax.legend(loc=3, bbox_to_anchor=(0, -0.5), frameon=False, numpoints=1)
    ax.legend(loc='best', frameon=False, numpoints=1)


def plot_signal_sum(ax, PS, fname='LFPsum.h5', unit='mV', scaling_factor=1.,
                    ylabels=True, scalebar=True, vlimround=None,
                    T=[800, 1000], ylim=[-100, 0], colors='k',
                    label='', transient=500, rasterized=False):
    '''
    on axes plot the summed LFP contributions

    Parameters
    ----------
    ax: matplotlib.axes.AxesSubplot
    PS: ParamsLFP object
        LFP-simulation parameters
    fname: str/np.ndarray
        path to h5 file or ndim=2 numpy.ndarray
    unit: str
        scalebar unit
    scaling_factor: float
        scaling factor
    ylabels: bool
        show labels on y-axis
    scalebar: bool
        show scalebar in plot
    vlimround: None/float
        override autoscaling of data and scalebar
    T: list
        [tstart, tstop], which timeinterval
    ylim: list of floats
        see plt.gca().set_ylim
    color: str/colorspec tuple
        color of shown lines
    label: str
        line labels


    Returns
    -------
    vlimround: float
        scalebar scaling factor, i.e., to match up plots

    '''
    if type(fname) == str and os.path.isfile(fname):
        with h5py.File(fname, 'r') as f:
            shape = f['data'].shape
            srate = f['srate'][()]
            if len(shape) > 2:
                # flatten all but last axis
                data = f['data'][()].reshape((-1, shape[-1]))
            else:
                data = f['data'][()]

            tvec = np.arange(data.shape[-1]) * 1000. / srate

            # for mean subtraction
            datameanaxis1 = data[:, tvec >= transient].mean(axis=1)
    elif type(fname) == np.ndarray and fname.ndim == 2:
        data = fname
        tvec = np.arange(data.shape[1]) * PS.dt_output
        datameanaxis1 = data[:, tvec >= transient].mean(axis=1)
    else:
        errmsg = 'type(fname)={} not str or numpy.ndarray'.format(type(fname))
        raise Exception(errmsg)

    # slice
    slica = (tvec <= T[1]) & (tvec >= T[0])
    data = data[:, slica]

    # subtract mean in each channel
    dataT = data.T - datameanaxis1
    data = dataT.T

    zvec = -np.arange(data.shape[0])
    vlim = abs(data).max()
    if vlimround is None:
        vlimround = 2.**np.round(np.log2(vlim)) / scaling_factor
    else:
        pass

    yticklabels = []
    yticks = []

    for i, z in enumerate(zvec):
        if i == 0:
            ax.plot(tvec[slica], data[i] / vlimround + z, lw=1.,
                    rasterized=rasterized, label=label, clip_on=False)
        else:
            ax.plot(tvec[slica], data[i] / vlimround + z, lw=1.,
                    rasterized=rasterized, clip_on=False)
        if i % 2 == 0:
            yticklabels.append('%i' % (i+1))
        else:
            yticklabels.append('')
        yticks.append(z)

    if scalebar:
        ax.plot([tvec[slica][-1]+np.diff(T)*0.02,
                 tvec[slica][-1]+np.diff(T)*0.02],
                [-1, -2], lw=2, color='k', clip_on=False)
        ax.text(tvec[slica][-1]+np.diff(T)*0.04, -1.5,
                '$2^{' + '{}'.format(int(np.log2(vlimround))
                                     ) + '}$ ' + '{0}'.format(unit),
                color='k', rotation='vertical',
                va='center')

    ax.axis(ax.axis('tight'))
    ax.yaxis.set_ticks(yticks)
    if ylabels:
        ax.yaxis.set_ticklabels(yticklabels)
        ax.set_ylabel('channel', labelpad=0.0)
    else:
        ax.yaxis.set_ticklabels([])

    remove_axis_junk(ax)

    ax.set_xticks(T)
    ax.set_xlabel(r'$t$ (ms)', labelpad=0.0)
    ax.set_ylim(ylim)

    return vlimround
