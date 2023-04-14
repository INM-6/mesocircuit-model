"""PyNEST Mesocircuit: Analysis Parameters
------------------------------------------

A dictionary with parameters defining the network analysis.

"""
import numpy as np

ana_dict = {
    # dtypes for reading ASCII files from NEST, file extension: .dat
    'read_nest_ascii_dtypes': {
        'spike_recorder': {'names': ('nodeid', 'time_ms'),
                           'formats': ('i4', 'f8')},
        'positions': {'names': ('nodeid', 'x-position_mm', 'y-position_mm'),
                      'formats': ('i4', 'f8', 'f8')}},
    # settings for writing .dat files similar to ASCII files from NEST
    'write_ascii': {
        'spike_recorder': {'sortby': 'time_ms',
                           'fmt': ['%d', '%.3f']},
        'positions': {'sortby': 'nodeid',
                      'fmt': ['%d', '%f', '%f']}},

    # whether to analyze only spike data within a center disc of 1mm2,
    # if False: all recorded data will be analyzed
    'extract_1mm2': False,

    # quantities to be computed during the preprocessing phase.
    # order is important because of dependencies
    'datatypes_preprocess': np.array(['positions',
                                      'sptrains',
                                      'sptrains_bintime',
                                      'sptrains_bintime_binspace',
                                      'neuron_count_binspace',
                                      'inst_rates_bintime_binspace',
                                      'pos_sorting_arrays',
                                      ]),

    # quantities to be computed during the statistics phase
    'datatypes_statistics': np.array(['FRs',
                                      'LVs',
                                      'CCs_distances',
                                      'PSDs',
                                      'CCfuncs_thalamic_pulses',
                                      ]),

    # temporal bin size (in ms)
    'binsize_time': 0.5,
    # spatial bin size (in mm), must lead to an even number of bins along each
    # axis
    'binsize_space': 0.1,

    # sorting axis for raster plot. Options are 'x', 'y' and None.
    'sorting_axis': 'x',

    # startup transient to discard for computing statistics (in ms).
    # a good choice for consistency is to choose it equal to the presimulation
    # time sim_dict['t_presim']
    't_transient': 1000.0,

    # number of neurons to compute correlation coefficients from.
    # if 'auto': the population size of the smallest population is taken.
    # if the given number is higher than the smallest population size, the
    # latter is also assumed.
    'ccs_num_neurons': 200,
    # time interval for computing correlation coefficients (in ms).
    # a good choice is equal to the refractory time
    'ccs_time_interval': 2.0,

    # number of data points used in each block for the FFT
    'psd_NFFT': 512,

    # parameters for __compute_cc_funcs_thalamus_center()
    # time lag (in ms)
    'cc_funcs_tau': 25.0,
    # number of bins along the diagonal, must be an even number.
    # if the given number is higher than the number of bins in one dimension,
    # the latter is assumed.
    'cc_funcs_nbins_diag': 14,
}
