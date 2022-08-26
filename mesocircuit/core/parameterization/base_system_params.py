"""PyNEST Mesocircuit: System Parameters
----------------------------------------

A dictionary with parameters defining machine configurations.

"""

import sys
import os


if 'HOSTNAME' in os.environ:
    partition = ('dc-cpu'
                 if os.environ['HOSTNAME'].rfind('jureca') > 0
                 else 'batch')
else:
    partition = None

# parameters have to be specified for each machine type individually
sys_dict = {
    # high-performance computing system using the SLURM workload manager
    'hpc': {
        # network simulation
        'network': {
            # partition, on JURECA DC the default is 'dc-cpu'
            'partition': partition,
            # number of compute nodes
            'num_nodes': 4,
            # number of MPI processes per node
            'num_mpi_per_node': 8,
            # number of threads per MPI process
            'local_num_threads': 16,
            # wall clock time
            'wall_clock_time': '00:30:00'},
        # analysis, plotting and analysis_and_plotting all use the same
        # configuration
        'analysis_and_plotting': {
            'partition': partition,
            'num_nodes': 1,
            'num_mpi_per_node': 12,
            'local_num_threads': 1,
            'wall_clock_time': '00:30:00'
        },
        'lfp_simulation': {
            'partition': partition,
            'num_nodes': 16,
            'num_mpi_per_node': 128,
            'local_num_threads': 1,  # not used
            'wall_clock_time': '04:00:00'
        },
        'lfp_plotting': {
            'partition': partition,
            'num_nodes': 1,
            'num_mpi_per_node': 1,
            'local_num_threads': 1,  # not used
            'wall_clock_time': '00:15:00'
        }
    },
    # laptop
    'local': {
        # per default, use as many threads as available (physical cores) for
        # network simulation and as many MPI process as possible for
        # analysis and plotting
        'network': {
            # number of MPI processes
            'num_mpi': 1,
            # number of threads per MPI process
            # if 'auto', the number of threads is set such that the total
            # number of virtual processes equals the number of physical cores
            'local_num_threads': 'auto'},
        'analysis_and_plotting': {
            # '$(nproc)' gives the number of available logical cores
            'num_mpi': ('$(sysctl -n hw.physicalcpu)'
                        if sys.platform == 'darwin' else '$(($(nproc) / 2))'),
        },
        'lfp_simulation': {
            'num_mpi': ('$(sysctl -n hw.physicalcpu)'
                        if sys.platform == 'darwin' else '$(($(nproc) / 2))'),
        },
        'lfp_plotting': {
            'num_mpi': 1
        }
    }
}
