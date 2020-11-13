"""PyNEST Mesocircuit: SpikeAnalysis Class
------------------------------------------

The SpikeAnalysis Class defines functions to preprocess spike activity and
compute statistics.

"""

import os
import warnings
import glob
import h5py
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from prettytable import PrettyTable
from ..helpers import parallelism_time as pt
from ..helpers import io

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class SpikeAnalysis:
    """ 
    Provides functions to analyze the spiking data written out by NEST.

    Instantiating a SpikeAnalysis object sets class attributes,
    merges spike and position files, changes node ids, and rescales spike times.
    The processed node ids start at 0 for each population.
    The pre-simulation is subtracted from all spike times.

    Parameters
    ---------
    sim_dict
        Dictionary containing all parameters specific to the simulation
        (derived from: ``base_sim_params.py``).
    net_dict
         Dictionary containing all parameters specific to the neuron and
         network models (derived from: ``base_network_params.py``).
    stim_dict
        Dictionary containing all parameters specific to the potential stimulus
        (derived from: ``base_stimulus_params.py``
    ana_dict
        Dictionary containing all parameters specific to the network analysis
        (derived from: ``base_analysis_params.py``

    """

    def __init__(self, sim_dict, net_dict, stim_dict, ana_dict):
        """
        Initializes some class attributes.
        """
        if RANK == 0:
            print('Instantiating a SpikeAnalysis object.')

        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.ana_dict = ana_dict

        # presynaptic population names
        self.X = self.net_dict['populations']
        if self.stim_dict['thalamic_input']:
            # thalamic population is treated as a cortical population
            self.X = np.append(self.X, self.stim_dict['th_name'])

        # postsynaptic population names
        self.Y = self.net_dict['populations']

        # population sizes
        self.N_X = self.net_dict['num_neurons']
        if self.stim_dict['thalamic_input']:
            self.N_X = np.append(self.N_X, self.stim_dict['num_th_neurons'])

        # temporal bins for pure and resampled spike trains
        self.time_bins = np.arange(
            0.,
            self.sim_dict['t_sim'],
            self.sim_dict['sim_resolution'])
        self.time_bins_rs = np.arange(
            0.,
            self.sim_dict['t_sim'],
            self.ana_dict['binsize_time'])

        # spatial bins
        # TODO check in later functions, old version used linspace
        self.space_bins = np.arange(-self.net_dict['extent'] / 2.,
                                    self.net_dict['extent'] / 2.,
                                    self.ana_dict['binsize_space'])
        return


    def preprocess_data(self):
        """
        Converts raw node ids to processed ones, merges raw spike and position
        files, prints a minimal sanity check of the data, performs basic
        preprocessing operations.

        New .dat files for plain spikes and positions are written and the main
        preprocessed data is stored in .h5 files.
        """
        if RANK == 0:
            print('Preprocessing data.')

        # load raw nodeids
        self.nodeids_raw = self.__load_raw_nodeids()

        # merge spike and position files generated on different threads or
        # processes
        num_spikes = pt.parallelize_by_array(self.X,
                                        self.__merge_raw_files_X,
                                        int,
                                        'spike_recorder')
        num_neurons = pt.parallelize_by_array(self.X,
                                        self.__merge_raw_files_X,
                                        int,
                                        'positions')
        if not (num_neurons == self.N_X).all():
            raise Exception ('Neuron numbers do not match.')

        # minimal analysis as sanity check
        self.__first_glance_at_data(num_spikes)

        # preprocess data of each population in parallel
        pt.parallelize_by_array(self.X,
                           self.__preprocess_data_X)
        return


    def compute_statistics(self):
        """
        Computes statistics in parallel for each population.
        """
        if RANK == 0:
            print('Computing statistics.')

        pt.parallelize_by_array(self.X,
                           self.__compute_statistics_X)
        return


    def merge_h5_files_populations(self):
        """
        Merges preprocessed data files and computed statistics for all
        populations.
        """
        if RANK == 0:
            print('Merging .h5 files for all populations.')

        pt.parallelize_by_array(self.ana_dict['datatypes_preprocess'],
                           self.__merge_h5_files_populations_datatype)

        pt.parallelize_by_array(self.ana_dict['datatypes_statistics'],
                           self.__merge_h5_files_populations_datatype)
        return


    def __load_raw_nodeids(self):
        """
        Loads raw node ids from file.

        Returns
        -------
        nodeids_raw
            Raw node ids: first and last id per population.
        """
        if RANK == 0:
            print('  Loading raw node ids.')

        # raw node ids: tuples of first and last id of each population;
        # only rank 0 reads from file and broadcasts the data
        if RANK == 0:
            nodeids_raw = np.loadtxt(os.path.join(self.sim_dict['path_raw_data'],
                                                  self.sim_dict['fname_nodeids']),
                                     dtype=int)
        else:
            nodeids_raw = None
        nodeids_raw = COMM.bcast(nodeids_raw, root=0)
        return nodeids_raw

    
    def __merge_raw_files_X(self, i, X, datatype):
        """
        Inner function to be used as argument of pt.parallelize_by_array()
        with array=self.X.
        Corresponding outer function: self.__preprocess_data()

        Processes raw NEST output files with file extention .dat.

        Raw NEST output files are loaded.
        Files are merged so that only one file per population exists, since
        there is typically one file per virtual process (as for spike files,
        datatype='spike_recorder') or per MPI process (as for position files,
        datatype='positions').
        Node ids and, if applicable also, spike times are processed.
        The processed node ids start at 0 for each population.
        The pre-simulation time is subtracted.
        The final processed data is written to file.

        Parameters
        ----------
        i
            Iterator of populations
            (to be set by outer parallel function).
        X
            Population names
            (to be set by outer parralel function).
        datatype
            Options are 'spike_recorder' and 'positions'.
            
        Returns
        -------
        num_rows
            An array with the number of rows in the final files.
            datatype = 'spike_recorder': number of spikes per population.
            datatype = 'positions': number of neurons per population
        """

        if i == 0:
            print('  Merging raw files:', datatype)

        # gather names of single files
        single_files = glob.glob(os.path.join(
            self.sim_dict['path_raw_data'],
            datatype + '_' + X + '*.dat'))

        # load data from single files and combine them
        read_dtype = self.ana_dict['read_nest_ascii_dtypes'][datatype]
        comb_data = np.array([[]], dtype=read_dtype)
        # skip three rows in raw nest output
        skiprows = 3 if datatype == 'spike_recorder' else 0
        for fn in single_files:
            # ignore all warnings of np.loadtxt(), target in particular
            # 'Empty input file'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data = np.loadtxt(fn, dtype=read_dtype, skiprows=skiprows)
            comb_data = np.append(comb_data, data)

        # change from raw to processed node ids:
        # subtract the first one of the raw ids in each population
        comb_data['nodeid'] -= self.nodeids_raw[i][0]

        if 'time_ms' in comb_data.dtype.names:
            # subtract the pre-simulation time
            comb_data['time_ms'] -= self.sim_dict['t_presim']

        # sort the final data
        comb_data = np.sort(
            comb_data, order=self.ana_dict['write_ascii'][datatype]['sortby'])

        # number of rows corresponds to
        # 'spike_recorder': number of spikes
        # 'positions': number of neurons
        num_rows = np.shape(comb_data)[0]

        # write processed file
        fn = os.path.join(
            self.sim_dict['path_processed_data'],
            datatype + '_' + X + '.dat')
        np.savetxt(fn, comb_data, delimiter='\t',
                   header='\t '.join(read_dtype['names']),
                   fmt=self.ana_dict['write_ascii'][datatype]['fmt'])
        return num_rows


    def __first_glance_at_data(self, num_spikes):
        """
        Prints a table offering a first glance on the data.

        Parameters
        ----------
        num_spikes
            An array of spike counts per population.
        """
        # compute firing rates
        rates = np.divide(num_spikes, self.N_X)

        # collect overview data
        dtype = {'names': ('population', 'num_neurons', 'rate_s-1'),
                 'formats': ('U4', 'i4', 'f4')}
        ov = np.zeros(shape=(len(self.X)), dtype=dtype)
        ov['population'] = self.X
        ov['num_neurons'] = self.N_X
        ov['rate_s-1'] = np.around(rates, decimals=3)

        # convert to pretty table for printing
        overview = PrettyTable(ov.dtype.names)
        for row in ov:
            overview.add_row(row)
        overview.align = 'r'

        if RANK == 0:
            print('\n  First glance at data:')
            print(overview, '\n')
        return


    def __preprocess_data_X(self, i, X):
        """
        Inner function to be used as argument of pt.parallelize_by_array()
        with array=self.X.
        Corresponding outer function: self.preprocess_data()

        Each function computing a dataset already writes it to .h5 file.

        Parameters
        ----------
        i
            Iterator of populations
            (to be set by outer parallel function).
        X
            Population names
            (to be set by outer parralel function).
        """
        
        # load plain spike data and positions
        data_load = []
        for datatype in self.ana_dict['read_nest_ascii_dtypes'].keys():
            fn = os.path.join(self.sim_dict['path_processed_data'],
                              datatype + '_' + X + '.dat')
            # ignore all warnings of np.loadtxt(), target in particular
            # 'Empty input file'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data = np.loadtxt(fn,
                    dtype=self.ana_dict['read_nest_ascii_dtypes'][datatype])
            data_load.append(data)
        spikes, positions = data_load

        # order is important! # TODO improve
        for datatype in self.ana_dict['datatypes_preprocess']:
            if i==0:
                print('  Processing: ' + datatype)

            # time binned spike trains
            if datatype == 'sptrains':
                dataset = self.__time_binned_sptrains_X(
                    X, spikes, self.time_bins, dtype=np.uint8)
                is_sparse = True
                dtype = None

            # position sorting arrays
            elif datatype == 'pos_sorting_arrays':
                dataset = self.__pos_sorting_array_X(
                    X, positions)
                is_sparse = False
                dtype = int

            self.__write_dataset_to_h5_X(X, datatype, dataset, is_sparse, dtype)
        return


    def __write_dataset_to_h5_X(self,
        X, datatype, dataset, is_sparse, dtype=None):
        """
        Writes sparse and non-sparse datasets for population X to .h5.

        Parameters
        ----------
         X
            Population name.
        datatype
            Name of the dataset.
        dataset
            The data itself.
        is_sparse
            Whether the data shall be written in sparse format.
        dtype
            dtype only needed for non-sparse datasets.
        """
        fn = os.path.join(self.sim_dict['path_processed_data'],
                          datatype + '_' + X + '.h5')
        f = h5py.File(fn, 'w')

        if is_sparse:
            # TODO why not everything in COO format?
            if type(dataset) == sp.coo_matrix:
                d = dataset
            else:
                d = dataset.tocoo()

            group = f.create_group(X)
            dset = group.create_dataset('data_row_col',
                                        data=np.c_[d.data, d.row, d.col],
                                        compression='gzip',
                                        compression_opts=2,
                                        maxshape = (None, None))
            dset = group.create_dataset('shape',
                                        data=d.shape,
                                        maxshape= (None,))            
        else:
            f.create_dataset(X,
                                data=dataset,
                                dtype=dtype,
                                compression='gzip',
                                compression_opts=2,
                                chunks=True,
                                shape=dataset.shape)
        f.flush()
        f.close()
        return


    def __time_binned_sptrains_X(self, X, spikes, time_bins, dtype):
        """
        Computes a histogram with ones for each spike.

        Parameters
        ----------
        X
            Population name.
        spikes
            Array of node ids and spike times.
        time_bins
            Time bins.
        dtype
            An integer dtype that fits the data.

        Returns
        -------
        sptrains
            Spike trains as Compressed Sparse Row matrix.
        """
        # if no spikes were recorded, return an empty sparse matrix
        i = np.where(self.X==X)[0][0] # TODO
        shape = (self.N_X[i], time_bins.size)

        if spikes.size == 0:
            sptrains = sp.coo_matrix(shape, dtype=dtype)
        else: 
            # time bins shifted by one bin as needed by np.digitize()
            dt = time_bins[1] - time_bins[0]
            time_bins_digi = np.r_[time_bins[1:], [time_bins[-1] + dt]]
            # indices of time bins to which each spike time belongs
            time_indices = np.digitize(spikes['time_ms'], time_bins_digi)

            # create COO matrix
            data = np.ones(spikes.size, dtype=dtype)
            sptrains = sp.coo_matrix(
                (data, (spikes['nodeid'], time_indices)),
                shape=shape, dtype=dtype)
        return sptrains


    def __pos_sorting_array_X(self, X, positions):
        """
        Computes an array with indices for sorting node ids according to the
        given sorting axis.
        
        Parameters
        ----------
        X
            Population name.
        positions
            Positions of population X.

        Returns
        -------
        argsort
            Sorting array.
        """
        if self.ana_dict['sorting_axis'] == 'x':
            pos_sorting_arrays = np.argsort(positions['x-position_mm'])
        elif self.ana_dict['sorting_axis'] == 'y':
            pos_sorting_arrays = np.argsort(positions['y-position_mm'])
        elif self.ana_dict['sorting_axis'] == None:
            pos_sorting_arrays = np.arange(positions.size) 
        else:
            raise Exception ("Sorting axis is not 'x', 'y' or None.")
        return pos_sorting_arrays


    def __compute_statistics_X(self, i, X):
        """
        Inner function to be used as argument of pt.parallelize_by_array()
        with array=self.X.
        Corresponding outer function: self.compute_statistics()

        Each function computing a dataset already writes it to .h5 file.

        Parameters
        ----------
        i
            Iterator of populations
            (to be set by outer parallel function).
        X
            Population names
            (to be set by outer parralel function).
        """

        # load preprocessed data
        d = {}
        for datatype in self.ana_dict['datatypes_preprocess']:
            datatype_X = datatype + '_' + X
            fn = os.path.join(self.sim_dict['path_processed_data'], datatype_X + '.h5')
            data = h5py.File(fn, 'r')
            # load .h5 files with sparse data to csr format
            if 'data_row_col' in data[X]:
                data = io.load_h5_to_sparse_X(X, data)             
            d.update({datatype + '_X': data})


        # order is important! # TODO improve
        for datatype in self.ana_dict['datatypes_statistics']:
            if i==0:
                print('  Computing: ' + datatype)

            if d['sptrains_X'].size == 0:
                dataset = np.array([])
            else:
                # per-neuron firing rates
                if datatype == 'rates':
                    dataset = self.__compute_rates(X, d['sptrains_X']) 

                # local coefficients of variation
                elif datatype == 'LVs':
                    dataset = self.__compute_lvs(X, d['sptrains_X'])

                # correlation coefficients
                elif datatype == 'CCs':
                    dataset = self.__compute_ccs(
                        X, d['sptrains_X'], self.sim_dict['sim_resolution'])

                # power spectral densities
                elif datatype == 'PSDs':
                    dataset = self.__compute_psds(
                        X, d['sptrains_X'], self.sim_dict['sim_resolution'])

            self.__write_dataset_to_h5_X(X, datatype, dataset, is_sparse=False)
        return


    def __compute_rates(self, X, sptrains_X):
        """
        Computes the firing rate of each neuron by dividing the spike count by
        the simulation time.
        
        Parameters
        ----------
        X
            Population name.
        sptrains_X
            Sptrains of population X in sparse csr format.

        """
        count = np.array(sptrains_X.sum(axis=1)).flatten()
        rates = count * 1.E3 / self.sim_dict['t_sim'] # in 1/s
        return rates


    def __compute_lvs(self, X, sptrains_X):
        """
        Computes local coefficients of variation from inter-spike intervals.

        This function was modified from https://github.com/NeuralEnsemble/elephant

        Calculate the measure of local variation LV for
        a sequence of time intervals between events.
        Given a vector v containing a sequence of intervals, the LV is
        defined as:
        .math $$ LV := \\frac{1}{N}\\sum_{i=1}^{N-1}
                    \\frac{3(isi_i-isi_{i+1})^2}
                            {(isi_i+isi_{i+1})^2} $$
        The LV is typically computed as a substitute for the classical
        coefficient of variation for sequences of events which include
        some (relatively slow) rate fluctuation.  As with the CV, LV=1 for
        a sequence of intervals generated by a Poisson process.

        Parameters
        ----------
        X
            Population name.
        sptrains_X
            Sptrains of population X in sparse csr format.

        References
        ----------
        ..[1] Shinomoto, S., Shima, K., & Tanji, J. (2003). Differences in spiking
        patterns among cortical neurons. Neural Computation, 15, 2823-2842

        """
        lvs = np.zeros(sptrains_X.shape[0])
        for i,sptrain in enumerate(sptrains_X):
            # inter-spike intervals of spike trains of individual neurons in 
            # units of time steps of sptrains_X
            # (for isis in units of ms or s, multiply with the time step)
            isi = np.diff(np.where(sptrain.toarray())[1])
        
            if isi.size < 2:
                lvs[i] = np.nan
            else:
                lvs[i] = 3. * (
                    np.power(np.diff(isi) / (isi[:-1] + isi[1:]), 2)).mean()
        return lvs


    def __compute_ccs(self, X, sptrains_X, binsize_time):
        """
        Computes Pearson correlation coefficients, excluding auto-correlations.

        Parameters
        ----------
        X
            Population name.
        sptrains_X
            Sptrains of population X in sparse csr format.
        binsize_time
            Temporal resolution of sptrains_X (in ms).
        """
        # TODO update when TC is added

        min_num_neurons = np.min(self.net_dict['num_neurons'])
        if self.ana_dict['ccs_num_neurons'] == 'auto' or \
            self.ana_dict['ccs_num_neurons'] > min_num_neurons:
            num_neurons = min_num_neurons
        else:
            num_neurons = self.ana_dict['ccs_num_neurons']

        # convert to array, remove non-spiking neurons, and extract at most
        # num_neurons neurons for computing CCs
        spt = sptrains_X.toarray()
        spt = spt[~np.all(spt==0, axis=1)]
        spt = spt[:num_neurons, :]
        num_neurons_spk = np.shape(spt)[0]

        if X=='L23E':
            print('    Using ' + str(num_neurons) + ' neurons in each ' +
                  'population for computing CCs (if no exception given).')
        if num_neurons != num_neurons_spk:
            print('    Exception: Computing CCs of ' + X + ' from ' + 
                  str(num_neurons_spk) + ' neurons because not all selected ' +
                  str(num_neurons) + ' neurons spiked.')
        
        # bin spike data according to given interval
        ntbin = int(self.ana_dict['ccs_time_interval'] / binsize_time)
        spt = spt.reshape(num_neurons_spk, -1, ntbin).sum(axis=-1)

        ccs = np.corrcoef(spt)

        # mask lower triangle: elements below the k-th diagonal are zeroed
        # (k=1 excludes auto-correlations, k=0 would include them)
        mask = np.triu(np.ones(ccs.shape), k=1).astype(bool)
        ccs = ccs[mask]
        return ccs


    def __compute_psds(self, X, sptrains_X, binsize_time):
        """
        Computes population-rate power spectral densities.

        Parameters
        ----------
        X
            Population name.
        sptrains_X
            Sptrains of population X in sparse csr format.
        binsize_time
            Temporal resolution of sptrains_X (in ms).
        """
        # sampling frequency
        Fs = 1000. / binsize_time
        # number of points of overlap between segments
        noverlap = int(self.ana_dict['psd_NFFT'] * 3/4)

        # detrend data
        x = np.array(sptrains_X.sum(axis=0), dtype=float).flatten()
        x -= x.mean()

        Pxx, freq = plt.psd(x, NFFT=self.ana_dict['psd_NFFT'],
                            Fs=Fs, noverlap=noverlap)
        # frequencies (in 1/s), PSDs (in s^{-2} / Hz)
        psds = np.array([freq, Pxx])
        return psds


    def __merge_h5_files_populations_datatype(self, i, datatype):
        """
        Inner function to be used as argument of pt.parallelize_by_array()
        with array=datatypes.
        Corresponding outer function: self.__preprocess_data()

        Parameters
        ----------
        i
            Iterator of datatypes
            (to be set by outer parallel function).
        datatype
            Datatype to merge file across populations
            (to be set by outer parralel function).
        """
        print('  Merging .h5 files: ' + datatype)

        fn = os.path.join(self.sim_dict['path_processed_data'],
                         'all_' + datatype + '.h5')

        f = h5py.File(fn, 'w')
        for X in self.X:
            fn_X = os.path.join(self.sim_dict['path_processed_data'],
                                datatype + '_' + X + '.h5')
            f_X = h5py.File(fn_X, 'r')
            f.copy(f_X[X], X)
            f_X.close()
            os.system('rm ' + fn_X)
        f.close()
        return

















