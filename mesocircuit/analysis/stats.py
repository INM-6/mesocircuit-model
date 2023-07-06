"""Generic statistics
---------------------

"""

import numpy as np
import matplotlib.pyplot as plt
from hybridLFPy import helperfun


def ztransform(x):
    '''
    Returns signal with zero mean and standard deviation of 1.
    If a signal with zero standard deviation is supplied, a zero vector
    is returned.

    Parameters
    ----------
    x: ndarray
        Signal.

    Returns
    -------
    ztrans: ndarray
        Z-transformed data.

    '''
    if x.std() == 0:
        ztrans = np.zeros_like(x)
    else:
        ztrans = (x - x.mean()) / x.std()
    return ztrans


def compute_psd(x, Fs, NFFT, detrend='mean', overlap=3 / 4):
    """
    Compute power sprectrum `Pxx` of signal `x` using
    matplotlib.mlab.psd function

    Parameters
    ----------
    x: ndarray
        1-D array or sequence.
    Fs: float
        Sampling frequency.
    NFFT: int
        Number of data points used in each block for the FFT
    detrend: {'none', 'mean', 'linear'} or callable, default 'mean'
        Detrend data before fft-ing.
    overlap: float
        Traction of NFFT points of overlap between segments.
    """
    noverlap = int(overlap * NFFT)
    return plt.mlab.psd(x, NFFT=NFFT, Fs=Fs,
                        detrend=detrend, noverlap=noverlap)


def pdist_pbc(X, extent=(1, 1), edge_wrap=False):
    '''Sort of clone of `scipy.spatial.distance.pdist(xy, metric='euclidean')`
    that supports periodic boundary conditions

    Parameters
    ----------
    xy: ndarray
        shape (n, 2) array with x- and y-positions
    extent: len 2 tuple of floats
        (x, y)-extent of boundary
    edge_wrap: bool
        if True, assume periodic boundary conditions. 
        If False [default], produce same output as
        `scipy.spatial.distance.pdist(xy, metric='euclidean')` 

    Returns
    -------
    Y: ndarray
        Returns a condensed distance matrix Y. For each :math:`i` and :math:`j`
        (where :math:`i<j<m`),where m is the number of original observations.
        The metric ``dist(u=X[i], v=X[j])`` is computed and stored in entry ``m
        * i + j - ((i + 2) * (i + 1)) // 2``.
    '''
    d_h = np.array([])
    for i in range(X.shape[0]):
        d_ = helperfun._calc_radial_dist_to_cell(
            x=X[i, 0],
            y=X[i, 1],
            Xpos=X[i+1:],
            xextent=extent[0],
            yextent=extent[1],
            edge_wrap=edge_wrap
        )
        d_h = np.r_[d_h, d_]
    return d_h