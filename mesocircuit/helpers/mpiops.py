"""MPI operations
-----------------

Gatherv implementation.

"""

import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def GathervRecordArray(data, op=MPI.COMM_WORLD.Gatherv, root=0):
    """
    Simplifies `MPI:Gatherv` for structured/record ndarrays with mixed data types
    (int/float types as supported by `MPI:Gatherv`)

    Parameters
    ----------
    data: structured/record ndarray
        Structured/record array data to be gathered
    op: func
        MPI:Gatherv
    root: int
        RANK number

    Returns
    -------
    recvbuf: structured ndarray or None
        if `op=='Gatherv'`, returns gathered data on RANK `root` and `None` on
        other RANKs.

    Examples
    --------
    2 MPI processes:
    >>> import numpy as np
    >>> from mpi4py import MPI
    >>> COMM = MPI.COMM_WORLD
    >>> RANK = COMM.Get_rank()
    >>> gids = np.arange(5) + RANK * 5
    >>> positions = (np.random.rand(gids.size * 2) - 0.5
    >>>              ).reshape((gids.size, -1))
    >>> names = ['gid', 'x', 'y']
    >>> formats = ['i8', 'f8', 'f8']
    >>> data = np.recarray((gids.size, ), names=names, formats=formats)
    >>> data.gid = gids
    >>> data.x = positions[:, 0]
    >>> data.y = positions[:, 1]
    >>> DATA = GathervRecordArray(data)
    >>> print('RANK: {}; DATA={}'.format(RANK, DATA))
    RANK: 1; DATA=None
    RANK: 0; DATA=[(0, -0.1665335 ,  0.12929303) (1, -0.43021416, -0.32771718)
     (2,  0.0251959 , -0.42943778) (3, -0.39318116, -0.21901001)
     (4,  0.1211439 ,  0.43867859) (5, -0.05488427, -0.02737131)
     (6,  0.15760976, -0.31423769) (7,  0.33918007, -0.33955388)
     (8,  0.25559554,  0.00316074) (9, -0.17828942,  0.2760958 )]
    """
    counts = np.array(COMM.gather(data.size, root))

    if RANK == 0:
        DATA = np.empty(counts.sum(), dtype=data.dtype)
    else:
        DATA = None

    for name, format in data.dtype.fields.items():
        if RANK == root:
            recvbuf = np.empty(counts.sum(), dtype=format[0])
        else:
            recvbuf = None

        op(sendbuf=np.array(data[name]), recvbuf=(recvbuf, counts), root=root)

        if RANK == root:
            DATA[name] = recvbuf

    return DATA


if __name__ == '__main__':

    gids = np.arange(5) + RANK * 5
    positions = (np.random.rand(gids.size * 2) - 0.5).reshape((gids.size, -1))

    names = ['gid', 'x', 'y']
    formats = ['i8', 'f8', 'f8']

    data = np.recarray((gids.size, ), names=names, formats=formats)
    data.gid = gids
    data.x = positions[:, 0]
    data.y = positions[:, 1]

    DATA = GathervRecordArray(data)

    print('RANK: {}; DATA={}'.format(RANK, DATA))
