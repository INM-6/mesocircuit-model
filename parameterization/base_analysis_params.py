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
}
