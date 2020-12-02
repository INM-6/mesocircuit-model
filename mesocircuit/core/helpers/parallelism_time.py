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

    print_times(filename, logtime_data)
    COMM.Barrier()


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
    # if not evenly divisible, num_its_rank * num_procs > num_its such that
    # the highest rank (= num_procs - 1) has less iterations to perform
    num_its_rank = np.ceil(num_its / num_procs).astype(int)

    times_local = np.zeros(num_its_rank)
    times_global = np.zeros(num_its_rank * SIZE)

    COMM.Barrier()
    if RANK < num_procs:
        for i, func in enumerate(funcs):
            if RANK == int(i / num_its_rank):
                start_time = time.time()

                if type(func) == list:
                    fun, args = func
                    fun(*args)
                else:
                    fun = func
                    fun()

                end_time = time.time()
                times_local[i % num_its_rank] += end_time - start_time

    # gather and concatenate MPI-local results
    COMM.Allgather(times_local, times_global)
    times = times_global[:num_its]
    COMM.Barrier()

    # set up logtime_data structure
    # extract all function names
    func_names = []
    for i, func in enumerate(funcs):
        if type(func) == list:
            fun = func[0]
        else:
            fun = func
        func_names.append(fun.__name__)
    func_names = np.array(func_names)

    logtime_data = []
    for i in np.arange(num_procs):
        # string with all functions executed on rank i
        func_names_rank = func_names[np.where(np.arange(num_procs) == int(i / num_its_rank))]
        func_names_rank = ','.join(func_names_rank)

        # times are 0 for all but rank i
        logtime = np.zeros(SIZE)
        logtime[i] = times[i]
        logtime_data.append([func_names_rank, list(logtime)])

    print_times(filename, logtime_data)


def print_times(filename, logtimes):
    '''
    Prints times measured on each MPI rank for each function in logtimes.

    Parameters
    ----------
    filename
        Name of the file which calls the timed functions, e.g., 'run_network.py'.
    logtimes
        List with time measurements recorded with timeit().
    '''
    if RANK != 0:
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
    # if not evenly divisible, num_its_rank * num_procs > num_its such that
    # the highest rank (= num_procs - 1) has less iterations to perform
    num_its_rank = np.ceil(num_its / num_procs).astype(int)

    res_local = np.zeros(num_its_rank, dtype=result_dtype)
    res_global = np.zeros(num_its_rank * SIZE, dtype=result_dtype)

    COMM.Barrier()
    if RANK < num_procs:
        for i, val in enumerate(array):
            if RANK == int(i / num_its_rank):
                res_local[i % num_its_rank] = func(i, val, *args)
    else:
        pass
    # gather and concatenate MPI-local results
    COMM.Allgather(res_local, res_global)
    result = res_global[:num_its]
    COMM.Barrier()
    return result
