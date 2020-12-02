import os
import numpy as np
import h5py
import scipy.sparse as sp


class BaseAnalysisPlotting:

    def __init__(self, sim_dict, net_dict, stim_dict, ana_dict):

        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.ana_dict = ana_dict

        # presynaptic populations
        self.X = self.net_dict['populations']
        if self.stim_dict['thalamic_input']:
            # thalamic population is treated as a cortical population
            self.X = np.append(self.X, self.stim_dict['th_name'])

        # postsynaptic populations
        self.Y = self.net_dict['populations']

        # population sizes
        self.N_X = self.net_dict['num_neurons']
        if self.stim_dict['thalamic_input']:
            self.N_X = np.append(self.N_X, self.stim_dict['num_th_neurons'])

        # temporal bins for raw and resampled spike trains
        self.time_bins_sim = np.arange(
            0.,
            self.sim_dict['t_sim'] + self.sim_dict['sim_resolution'],
            self.sim_dict['sim_resolution'])
        self.time_bins_rs = np.arange(
            0.,
            self.sim_dict['t_sim'] + self.ana_dict['binsize_time'],
            self.ana_dict['binsize_time'])

        # spatial bins
        self.space_bins = np.linspace(-self.net_dict['extent'] / 2.,
                                      self.net_dict['extent'] / 2.,
                                      int(self.net_dict['extent'] / \
                                          self.ana_dict['binsize_space'] + 1))
        return


    def write_dataset_to_h5_X(self,
        X, datatype, dataset, is_sparse, dataset_dtype=None):
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
        dataset_dtype
            dtype only needed for non-sparse datasets.
        """
        fn = os.path.join(self.sim_dict['path_processed_data'],
                          datatype + '_' + X + '.h5')
        f = h5py.File(fn, 'w')

        if is_sparse:
            if type(dataset) == sp.coo_matrix:
                d = dataset
            else:
                d = dataset.tocoo()

            group = f.create_group(X)
            group.create_dataset('data_row_col',
                                 data=np.c_[d.data, d.row, d.col],
                                 compression='gzip',
                                 compression_opts=2,
                                 maxshape = (None, None))
            group.create_dataset('shape',
                                 data=d.shape,
                                 maxshape= (None,))            
        else:
            if type(dataset) == dict:
                group = f.create_group(X)
                for key,value in dataset.items():
                    group.create_dataset(key,
                                         data=value,
                                         dtype=dataset_dtype,
                                         compression='gzip',
                                         compression_opts=2,
                                         chunks=True,
                                         shape=value.shape)


            else:
                f.create_dataset(X,
                                data=dataset,
                                dtype=dataset_dtype,
                                compression='gzip',
                                compression_opts=2,
                                chunks=True,
                                shape=dataset.shape)
        f.flush()
        f.close()
        return


    def load_h5_to_sparse_X(self, X, h5data):
        """
        Loads sparse matrix stored in COOrdinate format in HDF5.

        Parameters
        ----------
        X
            Group name for datasets
            'data', 'row', 'col' vectors of equal length
            'shape' : shape of array tuple
        h5data
            Open .h5 file.
        """
        data_X = sp.coo_matrix((h5data[X]['data_row_col'][()][:, 0],
                                (h5data[X]['data_row_col'][()][:, 1],
                                h5data[X]['data_row_col'][()][:, 2])),
                                shape=h5data[X]['shape'][()])
        return data_X.tocsr()