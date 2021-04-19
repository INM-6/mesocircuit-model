'''misc input/output file functions'''

import scipy.sparse as sp


def load_h5_to_sparse_X(X, h5data):
    """
    Loads sparse matrix stored in COOrdinate format in HDF5.

    Parameters
    ----------
    X
        Group name for datasets
        'data', 'row', 'col' vectors of equal length
        'shape' : shape of array tuple
    h5data
        Open .h5 file.
    """
    data_X = sp.coo_matrix((h5data[X]['data_row_col'][()][:, 0],
                            (h5data[X]['data_row_col'][()][:, 1],
                            h5data[X]['data_row_col'][()][:, 2])),
                           shape=h5data[X]['shape'][()])
    return data_X.tocsr()
