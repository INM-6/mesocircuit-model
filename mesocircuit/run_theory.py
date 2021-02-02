import core.plotting.figures as figures
import core.plotting.plotting as plotting
import os
import pickle
import numpy as np
import lif_meanfield_tools as lmt
ureg = lmt.ureg


def run_lmt(parameterview, paramspace_keys=[]):
    handled_paramsets = []
    for paramspace_key in parameterview.keys():
        if paramspace_keys != [] and paramspace_key not in paramspace_keys:
            pass
        else:
            for data_path, ps_id in parameterview[paramspace_key]:
                if ps_id in handled_paramsets:
                    pass
                else:
                    print('Computing theory with LIF Meanfield Tools.')

                    # parameter path
                    path_parameters = os.path.join(
                        data_path, 'parameters', ps_id)

                    # lmt network object
                    nw = lmt.Network(
                        network_params=os.path.join(
                            path_parameters, 'lmt_dict.yaml'),
                        analysis_params=os.path.join(
                            os.path.dirname(__file__),
                            'core/parameterization/lmt_analysis_params.yaml'))

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
                        with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
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

                    handled_paramsets.append(ps_id)
        return
