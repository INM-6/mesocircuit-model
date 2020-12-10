import time
import numpy as np
from prettytable import PrettyTable

from mpi4py import MPI
# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def run_parallel_functions_sequentially(funcs, filename):
    '''
    Runs parallelized functions one ofter the other and measures time.

    Parameters
    ----------
    funcs
        List of functions.
        If no arguments should be passed, list entries are just function names.
        If arguments should be passed, list entries are lists of which the first
        elements are function names and the second a list of arguments.
    filename
        Name of the file calling the function.
    '''
    logtime_data = []

    for func in funcs:

        times_global = np.zeros(SIZE)
        start_time = time.time()

        if type(func) == list:
            fun, args = func
            fun(*args)
        else:
            fun = func
            fun()
    
        end_time = time.time()
        times_local = np.array([end_time - start_time])

        COMM.Allgather(times_local, times_global)
        logtime_data.append([fun.__name__, times_global])

    COMM.Barrier()
        
    num_ranks = len(logtime_data[0][1]) 
    rows = num_ranks + 1 # +1 for header
    cols = len(logtime_data) + 1 # +1 for ranks
    matrix = np.zeros((rows, cols), dtype='object')
    matrix[0, 0] = ''
    for c in np.arange(cols):
        if c == 0: 
            matrix[1:, c] = ['rank ' + str(r) for r in np.arange(num_ranks)]
        else:
            matrix[0, c] = logtime_data[c - 1][0] # function name
            times = logtime_data[c - 1][1]
            matrix[1:,c] = [str(np.around(t, decimals=3)) for t in times]

    title = 'Time measurements in s: ' + filename
    print_table(matrix, title)
    
    COMM.Barrier()
    return


def run_serial_functions_in_parallel(funcs, filename):
    '''
    Runs independent serial functions in parallel and measures times.

    Parameters
    ----------
    funcs
        List of functions.
        If no arguments should be passed, list entries are just function names.
        If arguments should be passed, list entries are lists of which the first
        elements are function names and the second a list of arguments.
    filename
        Name of the file calling the function.
    '''
    # total number of iterations
    num_its = len(funcs)
    # at most as many MPI processes needed as iterations have to be done
    num_procs = np.min([SIZE, num_its]).astype(int)
    # number of iterations assigned to each rank;
    # Allgather requires equally sized chunks.
    # maximum number of iterations per rank
    max_its_rank = np.ceil(num_its / num_procs).astype(int)

    times_local = np.zeros(max_its_rank)
    times_global = np.zeros(max_its_rank * SIZE)   

    COMM.Barrier()
    if RANK < num_procs:
        idx_local = 0 # MPI-local index
        for i,func in enumerate(funcs):
            if RANK == int(i % num_procs):
                start_time = time.time()

                if type(func) == list:
                    fun, args = func
                    fun(*args)
                else:
                    fun = func
                    fun()
  
                end_time = time.time()
                times_local[idx_local] = end_time - start_time
                idx_local += 1

    # gather and concatenate MPI-local results
    COMM.Allgather(times_local, times_global)
    times = np.reshape(times_global, (-1, max_its_rank)).flatten(order='F')[:num_its]
    COMM.Barrier()

    # set up logtime_data structure
    # extract all function names
    func_names = []
    for i,func in enumerate(funcs):
        if type(func) == list:
            fun = func[0]
        else:
            fun = func
        func_names.append(fun.__name__)
    func_names = np.array(func_names)
    
    logtime_data = {}
    for rank in np.arange(num_procs):
        r = 'rank ' + str(rank) 
        logtime_data[r] = []
        for i,func_name in enumerate(func_names):
            if rank == int(i % num_procs):
                logtime_data[r].append([func_name, times[i]])

    print_times_serial(filename, logtime_data)
    return


def print_times_serial(filename, logtime_data):
    """
    Prints times measured on each MPI rank for each function in logtimes.

    Parameters
    ----------
    filename
        Name of the file which calls the timed functions, e.g., 'run_network.py'.
    logtimes
        List with time measurements recorded with timeit().
    """
    if RANK != 0:
        return

    rank_sum = {}
    for r in logtime_data:
        times = np.array(logtime_data[r])[:,1].astype(float)
        rank_sum[r] = np.sum(times)

    sep = '+' + '-' * 79

    string = '\n' + sep +'\n| Time measurements in s: ' + filename + '\n'
    for r in logtime_data: # ranks
        string += sep + '\n| ' + r + ': '
        string += str(np.around(rank_sum[r], decimals=3))
        for f in logtime_data[r]: # functions
            string += '\n|-- ' + f[0] + ': ' + str(np.around(f[1], decimals=3))
        string += '\n'
    string += sep + '\n'
    print(string)
    return


def parallelize_by_array(array, func, result_dtype=None, *args):
    '''
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
    '''
    # total number of iterations
    num_its = len(array)
    if num_its == 0:
        return None
    # at most as many MPI processes needed as iterations have to be done
    num_procs = np.min([SIZE, num_its]).astype(int)
    # number of iterations assigned to each rank;
    # Allgather requires equally sized chunks.
    # maximum number of iterations per rank
    max_its_rank = np.ceil(num_its / num_procs).astype(int)

    res_local = np.zeros(max_its_rank, dtype=result_dtype)
    res_global = np.zeros(max_its_rank * SIZE, dtype=result_dtype)

    COMM.Barrier()
    if RANK < num_procs:
        idx_local = 0 # MPI-local index
        for i,val in enumerate(array):
            if RANK == int(i % num_procs):
                res_local[idx_local] = func(i, val, *args)
                idx_local += 1
    else:
        pass
    # gather and concatenate MPI-local results
    COMM.Allgather(res_local, res_global)
    result = np.reshape(res_global, (-1, max_its_rank)).flatten(order='F')[:num_its]
    COMM.Barrier() 
    return result  


def print_table(matrix, title=''):
    """
    Prints a nice table.

    Parameters
    ----------
    matrix
        First row are headers. All entries are formatted strings.

    """
    if RANK != 0:
        return

    print(matrix)

    return string
