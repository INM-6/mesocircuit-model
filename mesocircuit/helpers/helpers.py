"""Helpers
----------
"""

import numpy as np
import json
import operator
import hashlib
import pickle
import parameters as ps


def merge_dictionaries(main_dict, new_dict, path=None):
    """
    Merges new dictionary recursively into main dictionary.

    Parameters
    ----------
    main_dict
        Main dictionary.
    new_dict
        New dictionary with entries to overwrite main_dict.
    path
        Path in case of nested dictionary.

    Returns
    -------
    main_dict
        Updated dictionary.
    """
    if path is None:
        path = []
    for key in new_dict:
        if key in main_dict:
            if isinstance(main_dict[key], dict) and isinstance(new_dict[key], dict):
                merge_dictionaries(
                    main_dict[key], new_dict[key], path + [str(key)])
            else:
                main_dict[key] = new_dict[key]
        else:
            main_dict[key] = new_dict[key]
    return main_dict


def sort_deep_dict(d):
    """
    Sorts arbitrarily deep dictionaries into tuples.

    Parameter
    ---------
    d : dict

    Returns
    -------
    x : list of tuples of tuples of tuples ...
    """
    x = sorted(iter(list(d.items())), key=operator.itemgetter(0))
    for i, (key, value) in enumerate(x):
        if isinstance(value, dict) or isinstance(value, ps.ParameterSet):
            y = sorted(iter(list(value.items())), key=operator.itemgetter(0))
            x[i] = (key, y)
            for j, (k, v) in enumerate(y):
                if isinstance(v, dict) or isinstance(v, ps.ParameterSet):
                    y[j] = (k, sort_deep_dict(v))
    return x


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder for numpy types.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def get_unique_id(d):
    """
    Creates a unique hash key for an input dictionary.

    Parameters
    ----------
    d : dict
        E.g., parameter dictionary.

    Returns
    -------
    key : str
        Hash key.
    """
    d_sorted = sort_deep_dict(d)
    string = pickle.dumps(d_sorted)
    key = hashlib.md5(string).hexdigest()
    return key
