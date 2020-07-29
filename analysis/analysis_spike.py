"""PyNEST Mesocircuit: Network Class
------------------------------------

Main file of the mesocircuit defining the ``Network`` class with functions to
build and simulate the network.

"""

import os
import glob
import h5py
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from prettytable import PrettyTable

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class SpikeAnalysis:
    """ 
    Provides functions to analyze the spiking data written out by NEST.

    Instantiating a SpikeAnalysis object sets class attributes,
    merges spike and position files, changes node ids, and rescales spike times.
    The processed node ids start at 1 and are continguous.

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
        if RANK == 0:
            print('Instantiating a SpikeAnalysis object.')

        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.ana_dict = ana_dict

        # dtypes
        self.dtypes = {
            'spike_detector': {'names': ('nodeid', 'time_ms'),
                               'formats': ('i4', 'f8')},
            'positions': {'names': ('nodeid', 'x-position_mm', 'y-position_mm'),
                          'formats': ('i4', 'f8', 'f8')}}

        # thalamic population 'TC' is treated as the cortical populations
        # presynaptic population names
        # TODO add TC properly
        self.X = self.net_dict['populations'] 
        #self.X = np.append(self.net_dict['populations'], 'TC')
        # postsynaptic population names
        self.Y = self.net_dict['populations']
        # population sizes
        self.N_X = self.net_dict['num_neurons']
        #self.N_X = np.append(self.net_dict['num_neurons', self.net_dict['num_neurons_th'])


        # TODO when merging raw files, better not subtract the transient!!!

        # temporal bins for raw and resampled spike trains
        self.time_bins = np.arange(
            self.sim_dict['t_presim'],
            self.sim_dict['t_presim'] + self.sim_dict['t_sim'],
            self.sim_dict['sim_resolution'])
        self.time_bins_rs = np.arange(
            self.sim_dict['t_presim'],
            self.sim_dict['t_presim'] + self.sim_dict['t_sim'],
            self.ana_dict['binsize_time'])

        # spatial bins
        # TODO check in later functions, old version used linspace
        self.pos_bins = np.arange(-self.net_dict['extent'] / 2.,
                                  self.net_dict['extent'] / 2.,
                                  self.ana_dict['binsize_space'])

        # TODO consider to move the following block out of __init__()
        # convert raw node ids to processed ones
        self.nodeids_raw, self.nodeids_proc = self.__raw_to_processed_nodeids()

        # merge spike and position files
        num_spikes = self.__merge_raw_files('spike_detector')
        num_neurons = self.__merge_raw_files('positions')
        # TODO maybe move somewhere else
        if not (num_neurons == self.N_X).all():
            raise Exception ('Neuron numbers do not match.')

        # minimal analysis as sanity check
        self.__first_glance_on_data(num_spikes)
        return


    def preprocess_data(self):
        """
        """
        # preprocess data of each population in parallel
        self.__parallelize(self.X,
                           self.__preprocess_data_X,
                           None)


        # combine population results
        self.__merge_preprocessed_data()


        # spike trains of each neuron

        # for each population
        #   read data from dat file or get it somewhere else
        #   compute spike trains
        #   write one file per population
        # combine population files


        # option 1: population parallel
        # option 2: parallel in quantities to be computed

        # needed quantities:
        # pos_sorting_arrays
        # sptrains
        ### above also for subsampled 1mm
        return
    

    def __preprocess_data_X(self, i, X):
        """
        Inner function for to be used as argument of self.__parallelize()
        with array=self.X.
        Corresponding outer function: self.preprocess_data()

        Parameters
        ----------
        i
            Iterator of populations
            (to be set by outer parallel function).
        X
            Population names
            (to be set by outer parralel function).

        Returns
        -------
        """
        
        # TODO maybe return tuples of filenames and datasets
                

        # load plain spike data and positions
        data_load = []
        for datatype in ['spike_detector', 'positions']:
            fn = os.path.join(self.sim_dict['path_processed_data'],
                              datatype + '_' + X + '.dat')
            data_load.append(np.loadtxt(fn, dtype=self.dtypes[datatype]))
        spikes, positions = data_load

        # append lists for each computed quantity
        # [datatype, dataset, sparse, dtype]
        # if sparse==True: dtype is ignored (set to None for convenience)
        data_write = []
        # time binned spike trains
        sptrains = self.__compute_time_binned_sptrains(
            X, spikes, self.time_bins, dtype=np.uint8)
        data_write.append(['sptrains', sptrains, True, None])

        # position sorting array
        # TODO specify dtype
        pos_sorting_arrays = self.__get_pos_sorting_array(X, positions)
        data_write.append(['pos_sorting_arrays', pos_sorting_arrays, False, int])


        # write datasets to h5 files
        self.__write_datasets_to_h5_X(X, data_write)


        return


    def __write_datasets_to_h5_X(self, X, data):
        """
        Writes datasets for population X to .h5.

        Parameters
        ----------
        X
            Population name.
        data
            List of lists, each of which is: [datatype, dataset, sparse, dtype].
        """
        for datatype, dataset, sparse, dtype in data:
            fn = os.path.join(self.sim_dict['path_processed_data'],
                              datatype + '_' + X + '.h5')
            f = h5py.File(fn, 'w')

            if sparse:
                print('Sparse writing not implemented, yet.')
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


    def __compute_time_binned_sptrains(self, X, spikes, time_bins, dtype):
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
        if spikes.size == 0:
            num_neurons_X = self.N_X[np.where(self.X==X)[0]]
            shape = (num_neurons_X, time_bins)
            return sp.csr_matrix(shape, dtype=dtype)
        
        # time bins shifted by one bin as needed by np.digitize()
        dt = time_bins[1] - time_bins[0]
        time_bins_digi = np.r_[time_bins[1:], [time_bins[-1] + dt]]
        # indices of time bins to which each spike time belongs
        time_indices = np.digitize(spikes['time_ms'], time_bins_digi)

        # create COO matrix
        data = np.ones(spikes.size, dtype=dtype)
        sptrains = sp.coo_matrix(
            (data, (spikes['nodeid'], time_indices)))
            #shape=shape, dtype=dtype) # TODO is this shape and dtype needed?
        return sptrains.tocsr()


    def __get_pos_sorting_array(self, X, positions):
        """
        Get an array with indices for sorting node ids according to the given
        sorting axis.
        
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
            argsort = np.argsort(positions['x-position_mm'])
        elif self.ana_dict['sorting_axis'] == 'y':
            argsort = np.argsort(positions['y-position_mm'])
        elif self.ana_dict['sorting_axis'] == None:
            argsort = np.arange(positions.size) 
        else:
            raise Exception ("Sorting axis is not 'x', 'y' or None.")
        return argsort


    def __merge_preprocessed_data(self):
        """
        TODO maybe parallelize here over quantities
        """
        return






    def compute_statistics(self):
        """
        """
        return


    def __parallelize(self, array, func, result_dtype, *args):
        """
        Uses MPI to parallelize a loop over an array evaluating a function in
        every loop iteration and returning a result obtained with Allgather.

        Parameters
        ----------
        array
            Array-like to iterate over.
        func
            Function to be evaluated in every loop iteration.
        result_dtype
            Dtype of the entries in the returned array.
        *args
            Further arguments to function.

        Returns
        -------
        result
            Array combining the results of the individual MPI processes.
        """
        # total number of iterations
        num_its = len(array)
        # at most as many MPI processes needed as iterations have to be done
        num_procs = np.min([SIZE, num_its]).astype(int)
        # number of iterations assigned to each rank;
        # Allgather requires equally sized chunks.
        # if not evenly divisible, num_its_rank * num_procs > num_its such that
        # the highest rank (= num_procs - 1) has less iterations to perform
        num_its_rank = int(np.ceil(num_its) / num_procs)

        res_local = np.zeros(num_its_rank, dtype=result_dtype)
        res_global = np.zeros(num_its_rank * num_procs, dtype=result_dtype)
        if RANK < num_procs:
            for i,val in enumerate(array):
                if RANK == int(i / num_its_rank):
                    res_local[i % num_its_rank] = func(i, val, *args)
        else:
            pass
        # gather and concatenate MPI-local results
        COMM.Allgather(res_local, res_global)
        result = res_global[:num_its]
        COMM.barrier() # TODO needed? 
        return result   


    def __raw_to_processed_nodeids(self):
        """
        Loads raw node ids from file, converts them, and writes out the
        processed node ids (always the first and last id per population).

        The processed node ids start at 1 and are contiguous.
        """
        if RANK == 0:
            print('  Converting raw node ids to processed ones.')

        # raw node ids: tuples of first and last id of each population;
        # only rank 0 reads from file and broadcasts the data
        if RANK == 0:
            nodeids_raw = np.loadtxt(os.path.join(self.sim_dict['path_raw_data'],
                                                  self.sim_dict['fname_nodeids']),
                                     dtype=int)
        else:
            nodeids_raw = None
        nodeids_raw = COMM.bcast(nodeids_raw, root=0)

        # processed node ids: tuples of new first and last id of each population
        # new ids start at 1 and are contiguous
        first_nodeids_proc = np.array(
            [1 + np.sum(self.net_dict['num_neurons'][:i]) \
                for i in np.arange(self.net_dict['num_pops'])]).astype(int)
        nodeids_proc = np.c_[first_nodeids_proc,
                             np.add(first_nodeids_proc,
                                    self.net_dict['num_neurons']) - 1]

        if RANK == 0:
            np.savetxt(os.path.join(self.sim_dict['path_processed_data'],
                                    self.sim_dict['fname_nodeids']),
                       nodeids_proc,
                       fmt='%d')
        return nodeids_raw, nodeids_proc


    def __merge_raw_files(self, datatype='spike_detector'):
        """
        Processes raw NEST output files.

        Raw NEST output files are loaded.
        Files are merged so that only one file per population exists, since
        there is typically one file per virtual process (as for spike files,
        datatype='spike_detector') or per MPI process (as for position files,
        datatype='positions').
        Node ids and, if applicable also, spike times are processed.
        The processed node ids start at 1 and are continguous.
        The final processed data is written to file.

        Parameters
        ----------
        datatype
            Options are 'spike_detector' and 'positions'.

        Returns
        -------
        num_rows
            An array with the number of rows in the final files.
            datatype = 'spike_detector': number of spikes per population.
            datatype = 'positions': number of neurons per population.

        """
        if RANK == 0:
            print('  Merging raw files: ' + datatype)

        # merge raw files in parallel
        num_rows = self.__parallelize(self.X,
                                      self.__merge_raw_files_X,
                                      int,
                                      datatype)
        return num_rows


    def __merge_raw_files_X(self, i, X, datatype):
        """
        Inner function to be used as argument of self.__parallelize()
        with array=self.X.
        Corresponding outer function: self.__merge_raw_files()

        Parameters
        ----------
        i
            Iterator of populations
            (to be set by outer parallel function).
        X
            Population names
            (to be set by outer parralel function).
        datatype
            Options are 'spike_detector' and 'positions'.
        sortby
            Name to sort the processed data by.
        fmt
            Format used for writing processed data to file.
            
        Returns
        -------
        num_rows
            Number of rows.
        """
        # datatype-dependent parameters
        if datatype == 'spike_detector':
            sortby = 'time_ms'
            fmt = ['%d', '%.3f']
        elif datatype == 'positions':
            sortby = 'nodeid'
            fmt = ['%d', '%f', '%f'] 

        # gather names of single files
        single_files = glob.glob(os.path.join(
            self.sim_dict['path_raw_data'],
            datatype + '_' + X + '*.dat'))

        # load data from single files and combine them
        comb_data = np.array([[]], dtype=self.dtypes[datatype])
        # skip three rows in raw nest output
        skiprows = 3 if datatype == 'spike_detector' else 0
        for fn in single_files:
            data = np.loadtxt(fn, dtype=self.dtypes[datatype],
                              skiprows=skiprows)
            comb_data = np.append(comb_data, data)

        # change from raw to processed node ids,
        # subtract the first one of the raw ids and add the first one
        # of the processed ids per population 
        comb_data['nodeid'] += \
            -self.nodeids_raw[i][0] + \
            self.nodeids_proc[i][0]

        # sort the final data
        comb_data = np.sort(comb_data, order=sortby)

        # number of rows corresponds to
        # 'spike_detector': number of spikes
        # 'positions': number of neurons
        num_rows = np.shape(comb_data)[0]

        # write processed file
        fn = os.path.join(
            self.sim_dict['path_processed_data'],
            datatype + '_' + X + '.dat')
        header = '\t '.join(self.dtypes[datatype]['names']) 
        np.savetxt(fn, comb_data, delimiter='\t',
                   header=header, fmt=fmt)
        return num_rows


    def __first_glance_on_data(self, num_spikes):
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
        dtype = {'names': ('population', 'num_neurons', 'rate_s-1', 'first id', 'last id'),
                 'formats': ('U4', 'i4', 'f4', 'i4', 'i4')}
        ov = np.zeros(shape=(len(self.X)), dtype=dtype)
        ov['population'] = self.X
        ov['num_neurons'] = self.N_X
        ov['rate_s-1'] = rates
        ov['first id'] = self.nodeids_proc[:,0]
        ov['last id'] = self.nodeids_proc[:,1]

        # convert to pretty table for printing
        overview = PrettyTable(ov.dtype.names)
        for row in ov:
            overview.add_row(row)
        overview.align = 'r'

        if RANK == 0:
            print('First glance on data:')
            print(overview)
        return
