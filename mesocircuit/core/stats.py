#!/usr/bin/env python
"""Generic statistics module"""
import numpy as np


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
