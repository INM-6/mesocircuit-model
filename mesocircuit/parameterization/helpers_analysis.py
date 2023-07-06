"""Helper functions analysis
----------------------------

Helper function for deriving dependent analysis parameters from base parameter
dictionaries.

"""

import numpy as np
import copy


def derive_dependent_parameters(net_dict, sim_dict, base_ana_dict):
    """
    Derives analysis parameters which depend on the network and anaylsis base
    parameters.

    Parameters
    ----------
    net_dict
        Dictionary with base network parameters.
    sim_dict
        Dictionary with base simulation parameters.
    base_ana_dict
        Dictionary with base analysis parameters.

    Returns
    -------
    ana_dict
        Dictionary containing base and derived analysis parameters.
    """

    ana_dict = copy.copy(base_ana_dict)

    # population sizes
    ana_dict['N_X'] = net_dict['num_neurons']

    # presynaptic populations
    ana_dict['X'] = net_dict['populations']
    # postsynaptic populations (without thalamus)
    ana_dict['Y'] = net_dict['populations'][:-1]

    # minimum time index for removal of startup transient
    ana_dict['min_time_index_sim'] = int(base_ana_dict['t_transient'] /
                                         sim_dict['sim_resolution'])
    ana_dict['min_time_index_rs'] = int(base_ana_dict['t_transient'] /
                                        base_ana_dict['binsize_time'])

    # time over which statistics are computed (without transient)
    ana_dict['time_statistics'] = \
        sim_dict['t_presim'] + \
        sim_dict['t_sim'] - \
        base_ana_dict['t_transient']

    return ana_dict


def get_time_bins(t_presim, t_sim, binsize):
    '''
    Computes temporal bins for raw or resampled spike trains.
    Pre-simulation and actual simulation times are combined.

    Parameters
    ----------
    t_presim
        Presimulation time (in ms).
    t_sim
        Simulation time (in ms).
    binsize
        Either sim_dict['sim_resolution'] for raw spike trains or
        ana_dict['binsize_time'] for resampled spike trains (in ms).

    Returns
    -------
    time_bins
        Array of time bins (in ms).
        Either time_bins_sim for raw spike trains or time_bins_rs for
        resampled spike trains.
    '''
    time_bins = np.arange(
        0.,
        t_presim + t_sim + binsize,
        binsize)

    return time_bins


def get_space_bins(extent, binsize_space):
    '''Computes spatial bins.

    Parameters
    ----------
    extent
        Side length (in mm) of square network.
    binsize space
        Spatial bin size (in mm).

    Returns
    -------
    space bins
        Array of spatial bins (in mm).
    '''
    space_bins = np.linspace(
        -extent / 2.,
        extent / 2.,
        int(extent / binsize_space + 1))

    return space_bins
