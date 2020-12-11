import time
import numpy as np

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

        if isinstance(func, list):
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

    # matrix for printing results of time measurement
    num_ranks = len(logtime_data[0][1])
    rows = num_ranks + 1  # +1 for header
    cols = len(logtime_data) + 1  # +1 for ranks
    matrix = np.zeros((rows, cols), dtype='object')
    matrix[0, 0] = ''
    for c in np.arange(cols):
        if c == 0:
            matrix[1:, c] = ['RANK ' + str(r) for r in np.arange(num_ranks)]
        else:
            matrix[0, c] = logtime_data[c - 1][0]  # function name
            times = logtime_data[c - 1][1]
            matrix[1:, c] = [str(np.around(t, decimals=3)) for t in times]

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
        idx_local = 0  # MPI-local index
        for i, func in enumerate(funcs):
            if RANK == int(i % num_procs):
                start_time = time.time()

                if isinstance(func, list):
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

    if RANK == 0:
        times = np.reshape(times_global,
                           (-1, max_its_rank)).flatten(order='F')[:num_its]

        # set up logtime_data structure
        # extract all function names
        func_names = []
        for i, func in enumerate(funcs):
            if isinstance(func, list):
                fun = func[0]
            else:
                fun = func
            func_names.append(fun.__name__)
        func_names = np.array(func_names)

        logtime_data = {}
        for rank in np.arange(num_procs):
            r = 'RANK ' + str(rank)
            logtime_data[r] = []
            for i, func_name in enumerate(func_names):
                if rank == int(i % num_procs):
                    logtime_data[r].append([func_name, times[i]])

        # matrix for printing results of time measurement
        title = 'Time measurements in s: ' + filename
        matrix = np.zeros((0, 2), dtype=object)

        for r in logtime_data:
            submatrix = np.zeros((len(logtime_data[r]) + 1, 2), dtype=object)
            submatrix[0, 0] = r
            rank_time_sum = np.sum(
                np.array(
                    logtime_data[r])[
                    :, 1].astype(float))
            submatrix[0, 1] = str(np.around(rank_time_sum, decimals=3))
            for i, (func_name, timed) in enumerate(logtime_data[r]):
                submatrix[i + 1][0] = func_name
                submatrix[i + 1][1] = str(np.around(timed, decimals=3))
            matrix = np.vstack((matrix, submatrix))

        print_table(matrix, title, with_header=False)

    COMM.Barrier()
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
        idx_local = 0  # MPI-local index
        for i, val in enumerate(array):
            if RANK == int(i % num_procs):
                res_local[idx_local] = func(i, val, *args)
                idx_local += 1
    else:
        pass
    # gather and concatenate MPI-local results
    COMM.Allgather(res_local, res_global)
    result = np.reshape(res_global,
                        (-1, max_its_rank)).flatten(order='F')[:num_its]
    COMM.Barrier()
    return result


def print_table(matrix, title=None, with_header=True):
    """
    Prints a nice table.

    Parameters
    ----------
    matrix
        First row are headers. All entries are formatted strings.
    title
        Title.
    do_print
        Whether to print the table.
    """
    if RANK != 0:
        return

    # lengths in each matrix entry and maximum of each column
    lengths = np.vectorize(len)(matrix)
    cols_max = np.max(lengths, axis=0)

    # separator line
    sep = '+'
    for m in cols_max:
        sep += '-' * (m + 2) + '+'
    sep += '\n'

    string = '\n' + sep
    # left-align title
    if title:
        space = ' ' * (np.sum(cols_max) + 3 * (len(cols_max) - 1) - len(title))
        string += '| ' + title + space + ' |\n' + sep
    for r, row in enumerate(matrix):
        for c, val in enumerate(row):
            space = ' ' * (cols_max[c] - len(val))
            if c == 0:  # left-align
                string += '| ' + val + space + ' '
            else:  # right-align
                string += '| ' + space + val + ' '
        string += '|\n'
        if r == 0 and with_header:
            string += sep
    string += sep

    print(string)
    return
