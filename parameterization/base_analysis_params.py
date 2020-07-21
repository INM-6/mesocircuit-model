"""PyNEST Mesocircuit: Analysis Parameters
------------------------------------------

A dictionary with parameters defining the network analysis.

"""

import os

ana_dict = {

    # machine to run the analysis on. Options are 'local' and 'jureca'.
    'computer': 'jureca',

    # number of compute nodes (ignored if computer is 'local')
    'num_nodes': 1,
    # number of MPI processes per compute node
    'num_mpi_per_node': 2,

    # temporal bin size (in ms)
    'binsize_time': 0.5,
    # spatial bin size (in mm)
    'binsize_space': 0.1,

    # sorting axis for raster plot. Options are 'x', 'y' and None.
    'sorting_axis': 'x',
}
