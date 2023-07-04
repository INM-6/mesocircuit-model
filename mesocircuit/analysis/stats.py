#!/usr/bin/env python
"""Generic statistics module"""
import numpy as np
import matplotlib.pyplot as plt


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
