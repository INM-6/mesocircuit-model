"""PyNEST Mesocircuit: Analysis Parameters
------------------------------------------

A dictionary with parameters defining the network analysis.

"""
import numpy as np

ana_dict = {

    # machine to run the analysis on. Options are 'local' and 'jureca'.
    'computer': 'jureca',
    # number of compute nodes (ignored if computer is 'local')
    'num_nodes': 1,
    # number of MPI processes per compute node
    # (parallelization across populations including thalamus suggests 9)
    'num_mpi_per_node': 9,
    # wall clock time (ignored if computer is 'local')
    'wallclock_time': '00:15:00',

    # dtypes for reading ASCII files from NEST, file extension: .dat
    'read_nest_ascii_dtypes': {
        'spike_recorder': {'names': ('nodeid', 'time_ms'),
                           'formats': ('i4', 'f8')},
        'positions': {'names': ('nodeid', 'x-position_mm', 'y-position_mm'),
                      'formats': ('i4', 'f8', 'f8')}},
    # settings for writing .dat files similar to ASCII files from NEST
    'write_ascii': {
        'spike_recorder': {'sortby': 'time_ms',
                           'fmt' : ['%d', '%.3f']},
        'positions': {'sortby': 'nodeid',
                      'fmt': ['%d', '%f', '%f']}},

    # quantities to be computed during the preprocessing phase.
    # order is important because of dependencies
    'datatypes_preprocess': np.array(['sptrains',
                                      'sptrains_bintime',
                                      'sptrains_bintime_binspace',
                                      'neuron_count_binspace',
                                      'inst_rates_bintime_binspace',
                                      'pos_sorting_arrays',
                                      ]),

    # quantities to be computed during the statistics phase
    'datatypes_statistics': np.array(['rates',
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

    # number of neurons to compute correlation coefficients from.
    # if 'auto': the population size of the smallest population is taken.
    # if the given number is higher than the smallest population size, the
    # latter is also assumed.
    'ccs_num_neurons': 500,
    # time interval for computing correlation coefficients (in ms)
    'ccs_time_interval': 5.,

    # number of data points used in each block for the FFT
    'psd_NFFT': 512,

    ## parameters for __compute_cc_funcs_thalamus_center()
    # time lag (in ms)
    'cc_funcs_tau': 25.0,
    # number of bins along the diagonal, must be an even number.
    # if the given number is higher than the number of bins in one dimension,
    # the latter is assumed.
    'cc_funcs_nbins_diag': 14,
}
