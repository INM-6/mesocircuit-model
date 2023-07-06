"""Input-output functions
-------------------------
"""
import numpy as np
import scipy.sparse as sp
import os
import h5py


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
    if sparsetype == 'coo':
        return data_X
    else:
        return getattr(data_X, f'to{sparsetype}')()


def write_dataset_to_h5_X(
        X,
        data_dir_circuit,
        datatype,
        dataset,
        is_sparse,
        dataset_dtype=None):
    """
    Writes sparse and non-sparse datasets for population X to .h5.

    Parameters
    ----------
    X
        Population name.
    data_dir_circuit
        Data directory of the circuit.
    datatype
        Name of the dataset.
    dataset
        The data itself.
    is_sparse
        Whether the data shall be written in sparse format.
    dataset_dtype
        dtype only needed for non-sparse datasets.
    """
    fn = os.path.join(data_dir_circuit, 'processed_data',
                      f'{datatype}_{X}.h5')
    f = h5py.File(fn, 'w')

    if is_sparse:
        if isinstance(dataset, sp.coo_matrix):
            d = dataset
        else:
            d = dataset.tocoo()

        group = f.create_group(X)
        group.create_dataset('data_row_col',
                             data=np.c_[d.data, d.row, d.col],
                             compression='gzip',
                             compression_opts=2,
                             maxshape=(None, None))
        group.create_dataset('shape',
                             data=d.shape,
                             maxshape=(None,))
    else:
        if isinstance(dataset, dict):
            group = f.create_group(X)
            for key, value in dataset.items():
                group.create_dataset(key,
                                     data=value,
                                     dtype=dataset_dtype,
                                     compression='gzip',
                                     compression_opts=2,
                                     chunks=True,
                                     shape=value.shape)

        else:
            f.create_dataset(X,
                             data=dataset,
                             dtype=dataset_dtype,
                             compression='gzip',
                             compression_opts=2,
                             chunks=True,
                             shape=dataset.shape)
    f.flush()
    f.close()
    return
