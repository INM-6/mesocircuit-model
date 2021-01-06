'''precompute MUA as spike rate per spatiotemporal bin'''
import os
import h5py
import numpy as np
from core.analysis.spike_analysis import SpikeAnalysis
from core.helpers.io import load_h5_to_sparse_X


def write_mua_file(sim_dict, net_dict, ana_dict,
                   PS, path_lfp_data, networkSim):
    '''
    Parameters
    ----------
    sim_dict:
    net_dict:
    ana_dict:
    PS: ParameterSet
        LFP model parameters
    path_lfp_data: path
        Path to LFP output folder
    networkSim: CachedTopoNetwork instance
    '''
    # need method SpikeAnalysis._time_and_space_binned_sptrains_X()
    sana = SpikeAnalysis(sim_dict, net_dict, ana_dict)

    # monkey patch spatial bin edges to match electrode grid
    sana.space_bins = PS.MUA_bin_edges

    MUA = None
    for X in PS.Y_MUA:
        positions = {
            'x-position_mm': networkSim.positions[X][:, 0],
            'y-position_mm': networkSim.positions[X][:, 1]
        }

        with h5py.File(os.path.join(sim_dict['path_processed_data'],
                                    'all_sptrains_bintime.h5'), 'r') as f:
            sptrains_bintime = load_h5_to_sparse_X(X, f)
        tmp = sana._time_and_space_binned_sptrains_X(
            X, positions, sptrains_bintime,
            dtype=np.uint16)
        if MUA is None:
            MUA = tmp
        else:
            MUA = MUA + tmp
    # convert from #spikes to #spikes / s
    MUA = MUA * 1000 / ana_dict['binsize_time']

    # write file
    with h5py.File(os.path.join(path_lfp_data, PS.MUAFile), 'w') as f:
        f['data'] = MUA.todense()
        f['srate'] = 1000 / ana_dict['binsize_time']
