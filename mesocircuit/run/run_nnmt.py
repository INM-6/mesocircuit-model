# TODO not functional yet

import os
import pickle
import numpy as np
import nnmt

def run_single_nnmt(paramspace_key, ps_id, data_dir=None):
    """
    Computes some theoretical quantities with NNMT for a single
    parameter set.

    NNMT is the Neuronal Network Meanfield Toolbox
    (https://github.com/INM-6/nnmt).

    TODO move to a more appropriate place

    Parameters
    ----------
    paramspace_key
        A key identifying a parameter space.
    ps_id
        A parameter space id.
    data_dir
        Absolute path to write data to.
    """
    from ..plotting import figures, plotting
    import nnmt

    print(f'Computing theory for {paramspace_key} - {ps_id}.')

    os.chdir(os.path.join(data_dir, paramspace_key, ps_id))

    # nnmt network object of type Microcircuit
    nw = nnmt.models.Microcircuit(
        network_params=os.path.join(
            'parameters', 'nnmt_dict.yaml'), analysis_params=os.path.join(
            'parameters', 'nnmt_ana_dict.yaml'))

    # working point for exponentially shape post synaptic currents
    wp = nnmt.lif.exp.working_point(nw)
    # transfer function
    nnmt.lif.exp.transfer_function(nw)
    # delay distribution matrix
    nnmt.network_properties.delay_dist_matrix(nw)
    # effective connectivity matrix
    nnmt.lif.exp.effective_connectivity(nw)
    # power spectra
    power = nnmt.lif.exp.power_spectra(nw)
    freqs = nw.analysis_params['omegas'] / (2. * np.pi)

    # sensitivity measure
    sensitivity_dict = nnmt.lif.exp.sensitivity_measure_all_eigenmodes(nw)

    # corresponding plotting class
    dics = []
    for dic in ['sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
        with open(f'parameters/{dic}.pkl', 'rb') as f:
            dics.append(pickle.load(f))
    sim_dict, net_dict, ana_dict, plot_dict = dics

    pl = plotting.Plotting(
        sim_dict, net_dict, ana_dict, plot_dict)

    # overview figure
    figures.theory_overview(
        plot=pl,
        working_point=wp,
        frequencies=freqs,
        power=power,
        sensitivity=sensitivity_dict)

    return
