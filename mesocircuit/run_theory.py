import core.helpers.mesocircuit_framework as mesoframe
import core.plotting.figures as figures
import core.plotting.plotting as plotting
import os
import glob
import pickle
import numpy as np
import lif_meanfield_tools as lmt
ureg = lmt.ureg


def run_lmt(
        paramspace_keys=[], with_base_params=False,
        data_dir=mesoframe.auto_data_directory()):
    """
    Computes some theoretical quantities with LIF Meanfield Tools for all given
    parameter spaces.

    Parameters
    ----------
    paramspace_keys
        List of keys of parameter spaces defined in custom_ps_dicts.
        Providing an empty list means that all keys are evaluated (default=[]).
    with_base_params
        Whether to include a parameter space with only base parameters
        (default=False).
    data_dir
        Absolute path to write data to.
    """
    ps_keys = paramspace_keys
    if with_base_params:
        ps_keys.append('base')

    # parameters spaces identified by key
    for ps_key in ps_keys:
        full_data_paths = glob.glob(os.path.join(data_dir, ps_key, '*'))
        # parameter sets identified by ps_id
        for full_data_path in full_data_paths:
            os.chdir(full_data_path)
            ps_id = os.path.basename(full_data_path)
            print(f'Computing theory for {ps_key} - {ps_id}.')

            # lmt network object
            nw = lmt.Network(
                network_params=os.path.join(
                    'parameters', 'lmt_dict.yaml'), analysis_params=os.path.join(
                    'parameters', 'lmt_ana_dict.yaml'))

            # working point
            wp = nw.working_point()

            # power spectrum
            power = nw.power_spectra()
            freqs = nw.analysis_params['omegas'] / (2. * np.pi)

            # sensitivity measure
            pop_idx, freq_idx = np.unravel_index(np.argmax(power),
                                                 np.shape(power))
            frequency = freqs[freq_idx]

            sm = nw.sensitivity_measure(freq=frequency)
            eigs = nw.eigenvalue_spectra('MH')
            eigc = eigs[pop_idx][np.argmin(abs(eigs[pop_idx] - 1))]

            Z = nw.sensitivity_measure(frequency)
            k = np.asarray([1, 0]) - np.asarray([eigc.real, eigc.imag])
            k /= np.sqrt(np.dot(k, k))
            k_per = np.asarray([-k[1], k[0]])
            k_per /= np.sqrt(np.dot(k_per, k_per))
            Z_amp = Z.real * k[0] + Z.imag * k[1]
            Z_freq = Z.real * k_per[0] + Z.imag * k_per[1]

            # corresponding plotting cass
            dics = []
            for dic in [
                'sim_dict',
                'net_dict',
                'ana_dict',
                    'plot_dict']:
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
                sensitvity_amplitude=Z_amp,
                sensitivity_frequency=Z_freq,
                sensitivity_popidx_freq=[
                    pop_idx,
                    frequency])

        return
