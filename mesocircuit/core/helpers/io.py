'''misc input/output file functions'''

import scipy.sparse as sp


def load_h5_to_sparse_X(X, h5data, sparsetype='csr'):
    """
    Loads sparse matrix stored in COOrdinate format in HDF5.

    Parameters
    ----------
    X:
        Group name for datasets
        'data', 'row', 'col' vectors of equal length
        'shape' : shape of array tuple
    h5data: 
        Open .h5 file.
    sparsetype: str
        scipy.sparse matrix type in 
        ['csr', 'array', 'bsr', 'coo', 'csc', 'csr', 
         'dense', 'dia', 'dok', 'lil']. 
        Default is 'csr'
    
    Returns
    -------
    scipy.sparse matrix
    """
    assert sparsetype in ['csr', 'array', 'bsr', 'coo', 'csc', 'csr', 'dense', 
                    'dia', 'dok', 'lil'], f'type {sparsetype} not supported'
    data_X = sp.coo_matrix((h5data[X]['data_row_col'][()][:, 0],
                            (h5data[X]['data_row_col'][()][:, 1],
                             h5data[X]['data_row_col'][()][:, 2])),
                           shape=h5data[X]['shape'][()])
    if type == 'csr':
        return data_X
    else:
        return getattr(data_X, f'to{type}')()
