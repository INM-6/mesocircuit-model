import numpy as np
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