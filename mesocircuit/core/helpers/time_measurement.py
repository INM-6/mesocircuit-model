import time
import numpy as np
from prettytable import PrettyTable

from mpi4py import MPI
# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def timeit(method):
    '''
    Decorator for MPI-parallel functions to track time.
    
    Functions to be timed should have **kwargs as argument in the definition.
    When called, a list should be passed as keyword "logtime".
    The method name and the times on different ranks will be appended to that
    list.

    '''
    time_ranks = np.zeros(SIZE)
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        time_rank = np.array([end_time - start_time])
        COMM.Allgather(time_rank, time_ranks)
        if 'logtime' in kw:
            kw['logtime'].append([method.__name__, time_ranks])
        COMM.Barrier
        return result
    return timed


def timeit_serial(method):
    '''
    TODO needs improvement
    '''
    time_ranks = np.zeros(SIZE)
    def timed(*args, **kw):
        if kw['counter'] % SIZE != RANK:
            return

        start_time = time.time()
        method(*args, **kw)
        end_time = time.time()
        time_ranks[RANK] = end_time - start_time

        if 'logtime' in kw:
            kw['logtime'].append([method.__name__, time_ranks])
        return
    return timed


def print_times(filename, logtimes, rank=0):
    '''
    Prints times measured on each MPI rank for each function in logtimes.

    Parameters
    ----------
    filename
        Name of the file which calls the timed functions, e.g., 'run_network.py'.
    logtimes
        List with time measurements recorded with timeit().
    rank
        Rank used for printing.
    '''
    if RANK != rank:
        return
    
    x = PrettyTable()
    x.field_names = [''] + [logtime[0] for logtime in logtimes]

    for r in np.arange(len(logtimes[0][1])):
        row = ['rank ' + str(r)] + \
              [np.around(logtime[1][r], decimals=3) for logtime in logtimes]
        x.add_row(row)

    x.align = 'r'

    print('\nTime measurements in s: ' + filename)
    print(x, '\n')
    return