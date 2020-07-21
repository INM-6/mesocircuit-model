from mpi4py import MPI
import time
time_start = time.time()

# initialize MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# TODO




time_stop = time.time()

print(
    '\nTimes of Rank {}:\n'.format(RANK) +
    '  Total analysis time:  {:.3f} s\n'.format(
        time_stop - time_start))