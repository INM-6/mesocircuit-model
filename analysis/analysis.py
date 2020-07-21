"""PyNEST Mesocircuit: Network Class
------------------------------------

Main file of the mesocircuit defining the ``Network`` class with functions to
build and simulate the network.

"""

import os
import numpy as np
from mpi4py import MPI

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class Analysis:
    """ 
    TODO
    Instantiating a NetworkAnalysis object. 

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
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.ana_dict = ana_dict

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

        # temporal bins for raw and resampled spike trains
        self.time_bins = np.arange(self.sim_dict['t_presim'],
                                   self.sim_dict['t_sim'],
                                   self.sim_dict['sim_resolution'])
        self.time_bins_rs = np.arange(self.sim_dict['t_presim'],
                                      self.sim_dict['t_sim'],
                                      self.ana_dict['binsize_time'])

        # spatial bins
        # TODO check, old version used linspace
        self.pos_bins = np.arange(-self.net_dict['extent'] / 2.,
                                  self.net_dict['extent'] / 2.,
                                  self.ana_dict['binsize_space'])

        # raw node ids: tuples of first and last id of each population;
        # only rank 0 reads from file and broadcasts the data
        if RANK == 0:
            nodeids_raw = np.loadtxt(os.path.join(self.sim_dict['path_raw_data'],
                                                  self.sim_dict['fname_nodeids']),
                                     dtype=int)
        else:
            nodeids_raw = None
        self.nodeids_raw = COMM.bcast(nodeids_raw, root=0)

        # processed node ids: tuples of new first id and population size;
        # new ids start at 1 and are contiguous
        first_nodeids_proc = np.array(
            [1 + np.sum(self.net_dict['num_neurons'][:i]) \
                for i in np.arange(self.net_dict['num_pops'])]).astype(int)
        self.nodeids_proc = np.c_[first_nodeids_proc,
                                  self.net_dict['num_neurons']]

        if RANK == 0:
            np.savetxt(os.path.join(self.sim_dict['path_processed_data'],
                                    self.sim_dict['fname_nodeids']),
                       self.nodeids_proc,
                       fmt='%d')
