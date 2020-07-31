"""PyNEST Mesocircuit: Plotting Class
-------------------------------------

The Plotting Class defines plotting functions.

"""

import os
import h5py
import numpy as np
from mpi4py import MPI
import matplotlib
if not 'DISPLAY' in list(os.environ.keys()):
    matplotlib.use('Agg')
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
        matplotlib.rcParams.update(self.plot_dict['rcParams'])

        # load data of all datatypes;
        # files need to be closed in the end 
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

        gs = gridspec.GridSpec(1,1)




        self.__savefig('raster')

        return


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


