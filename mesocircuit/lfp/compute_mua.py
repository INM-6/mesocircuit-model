'''precompute MUA as spike rate per spatiotemporal bin'''
import os
import h5py
import numpy as np
import mesocircuit.analysis.spike_analysis as sana
from mesocircuit.helpers.io import load_h5_to_sparse_X


def write_mua_file(circuit,
                   PS, path_lfp_data, networkSim):
    '''
    Parameters
    ----------
    circuit:
    PS: ParameterSet
        LFP model parameters
    path_lfp_data: path
        Path to LFP output folder
    networkSim: CachedTopoNetwork instance
    '''

    MUA = None
    for X in PS.Y_MUA:
        positions = {
            'x-position_mm': networkSim.positions[X][:, 0],
            'y-position_mm': networkSim.positions[X][:, 1]
        }

        with h5py.File(os.path.join(os.path.split(path_lfp_data)[0],
                                    'processed_data',
                                    'all_sptrains_bintime.h5'), 'r') as f:
            sptrains_bintime = load_h5_to_sparse_X(
                X=X, h5data=f, sparsetype='coo')
        tmp = sana._time_and_space_binned_sptrains_X(
            positions=positions,
            sptrains_bintime=sptrains_bintime,
            # monkey patch spatial bin edges to match electrode grid
            space_bins=PS.MUA_bin_edges,
            dtype=np.uint16)
        if MUA is None:
            MUA = tmp
        else:
            MUA = MUA + tmp
    # convert from #spikes to #spikes / s
    MUA = MUA * 1000 / circuit.ana_dict['binsize_time']

    # write file
    with h5py.File(os.path.join(path_lfp_data, PS.MUAFile), 'w') as f:
        f['data'] = MUA.todense()
        f['srate'] = 1000 / circuit.ana_dict['binsize_time']
