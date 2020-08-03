"""PyNEST Mesocircuit: Plotting Class
-------------------------------------

The Plotting Class defines plotting functions.

"""

import os
import h5py
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
import matplotlib as mpl
if not 'DISPLAY' in list(os.environ.keys()):
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class Plotting:
    """ 
    Provides functions to plot the analyzed data.

    All functions that create a figure start with 'fig_'.

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
    plot_dict
        Dictionary containing all parameters specific to the plotting
        (derived from: ``base_plotting_params.py``

    """

    def __init__(self, sim_dict, net_dict, stim_dict, ana_dict, plot_dict):
        """
        Initializes some class attributes.
        """
        if RANK == 0:
            print('Instantiating a Plotting object.')

        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.ana_dict = ana_dict
        self.plot_dict = plot_dict

        # update the matplotlib.rcParams
        mpl.rcParams.update(self.plot_dict['rcParams'])

        # TODO this is currently the same as in the __init__ of analysis
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

        # load data of all datatypes;
        # files need to be closed in the end
        # TODO only temporary
        for datatype in np.append(self.ana_dict['datatypes_preprocess'],
                                  self.ana_dict['datatypes_statistics']):
            all_datatype = 'all_' + datatype
            data = self.__load_h5(all_datatype)
            globals().update({all_datatype: data})

        return


    def fig_raster(self):
        """
        """
        fig = plt.figure(figsize=(self.plot_dict['fig_width_1col'], 5.))

        fig.subplots_adjust(top=0.98, bottom=0.1, left=0.17, right=0.92)
        gs = gridspec.GridSpec(1, 1)
        self.__plot_raster_sorted(
            gs[0,0],
            self.X,
            all_sptrains,
            all_pos_sorting_arrays,
            self.sim_dict['sim_resolution'],
            self.plot_dict['time_interval_short'])

        self.__savefig('raster')
        return


    def __plot_raster_sorted(self,
        gs,
        populations,
        all_sptrains_h5,
        all_pos_sorting_arrays_h5,
        time_step,
        time_interval,
        xlabels=True,
        ylabels=True):
        """
        Parameters
        ----------
        gs
            A gridspec cell to plot into.
        populations
        all_sptrains_h5
        all_pos_sorting_arrays_h5
        time_step
        time_interval
        xlabels
        ylabels
        """
        nums_shown = []
        yticks = []
        ax = plt.subplot(gs)   
        for i,X in enumerate(populations):
            data = self.__load_h5_to_sparse_X(X, all_sptrains_h5)

            # slice according to time interval
            time_indices = np.arange(
                time_interval[0] / time_step,
                time_interval[1] / time_step).astype(int)
            data = data[:, time_indices]

            # sort according to spatial axis
            space_indices = all_pos_sorting_arrays_h5[X][()]
            data = data[space_indices, :]

            # TODO subsampling based on neuron numbers, spike count, time interval?
            #bool_show = np.ones(num_neurons, dtype=bool)

            # number of neurons
            # (if subsampled data is used, this is unequal to the total number
            # of neurons in the network)
            num_neurons = all_pos_sorting_arrays_h5[X].shape[0]

            # get x,y indices and plot
            y, x = np.nonzero(data.toarray())
            ax.plot(x * time_step + time_interval[0],
                    -(np.sum(nums_shown) + y),
                    marker='$.$',
                    markersize=mpl.rcParams['lines.markersize']*0.25,
                    color=self.plot_dict['pop_colors'][i],
                    markeredgecolor='none',
                    linestyle='',
                    rasterized=True)
            nums_shown.append(num_neurons)
            yticks.append(-np.sum(nums_shown) + 0.5 * nums_shown[-1])

        # draw lines to separate poppulations on top
        for i,X in enumerate(populations[:-1]):
            ax.plot(time_interval, [-np.sum(nums_shown[:i+1])]*2,
                    'k',
                    linewidth=mpl.rcParams['axes.linewidth'])

        ax.set_xlim(time_interval[0], time_interval[1])
        ax.set_ylim(-np.sum(nums_shown), 0)

        ax.set_yticks(yticks)

        if xlabels:
            ax.set_xlabel('time (ms)')
        else:
            ax.set_xticklabels([])
        if ylabels:
            ax.set_yticklabels(self.plot_dict['pop_labels'][:len(nums_shown)])
        else:
            ax.set_yticklabels([])
        return ax

    
    def __add_label(self):
        """
        """
        return


    def __load_h5_to_sparse_X(self, X, h5data):
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


    def __load_h5(self, all_datatype):
        """
        Loads the data of a given datatype from .h5.

        Returned file needs to be closed manually.

        Parameters
        ----------
        datatype
            Options for the all_datatype are 'all_' combined with entries in
            ana_dict['datatypes_preprocess'] and in
            ana_dict['datatypes_statistics'].

        Returns
        -------
        data
            Open .h5 file.
        """
        fn = os.path.join(self.sim_dict['path_processed_data'],
                          all_datatype + '.h5')
        data = h5py.File(fn, 'r')
        return data


    def __savefig(self, filename):
        """
        Saves the current figure to .eps format.
        TODO: conversion for rasterized

        Parameters
        ----------
        filename
            Name of the file.
        """
        fn = os.path.join(self.sim_dict['path_plots'], filename + '.pdf')
        plt.savefig(fn)
        plt.close()
        return


