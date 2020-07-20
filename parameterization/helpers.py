"""PyNEST Mesocircuit: Helper Functions
---------------------------------------

General helper functions.

"""

import numpy as np
import parameters as ps
import os
import sys
import operator
import pickle
import hashlib

from . import helpers_network_stimulus as helpnet

# default parameters
from .base_sim_params import sim_dict
from .base_network_params import net_dict
from .base_stimulus_params import stim_dict


def evaluate_parameterspaces(
    filename='', paramspace_keys=[], with_base_params=False):
    """
    Evaluates the parameter spaces as specified by the arguments.


    Parameters
    ----------
    filename
        Name of the file <filename>.py defining the parameter spaces to be
        evaluated (without file extension). The parameters paces need to be in a
        dictionary named 'ps_dicts'.
        If no filename is given, i.e., filename='', parameters are evaluated
        according to the base parameters (default='').
    paramspace_keys
        List of keys of parameter spaces defined in <filename>.py. Providing an
        empty list means that all keys are evaluated (default=[]).
    with_base_params
        Whether to include a parameter space with only base parameters
        (default=False).

    Returns
    -------
    parameterview
        Dictionary as overview. Keys are names of parameter spaces and values
        are lists of parameter set ids.

    """
    
    ps_dicts = {}
    try:
        # TODO generalize folder name
        f = __import__('parameterization.' + filename, fromlist=['ps_dicts'])
        ps_dicts.update(f.ps_dicts)
    except:
        print('No parameterspaces read from file.')
    if with_base_params:
        ps_dicts.update({'base': {}})

    # parameterspaces built with the parameters module and indexed by
    # paramspace_key
    parameterspaces = {}
    # collection of unique parametersets indexed by ps_id
    parametersets = {}
    # overview of parameterspaces and corresponding ps_ids
    parameterview = {}

    for paramspace_key in sorted(ps_dicts):
        if (len(paramspace_keys)==0 or # all keys
            paramspace_key in paramspace_keys or # selected key(s)
            paramspace_key=='base'): # base parameters if with_base_params
            print(paramspace_key)
            parameterview[paramspace_key]= []

            parameterspaces[paramspace_key] = ps.ParameterSpace({})
            # start with default parameters and update
            for dic,vdic in zip(
                ['sim_dict', 'net_dict', 'stim_dict'],
                [sim_dict, net_dict, stim_dict]):
                parameterspaces[paramspace_key][dic] = dict(vdic) # copy is needed
                if dic in ps_dicts[paramspace_key]:
                    parameterspaces[paramspace_key][dic].update(
                    ps_dicts[paramspace_key][dic])

            for paramset in parameterspaces[paramspace_key].iter_inner():
                ps_id = get_unique_id(paramset)
                if ps_id in sorted(parametersets):
                    print('Skipping {0}, already in job list.'.format(ps_id))
                    pass
                else:
                    print(ps_id)
                    parametersets[ps_id] = paramset
                    parameterview[paramspace_key].append([
                        paramset['sim_dict']['data_path'], ps_id])

                    evaluate_parameterset(ps_id, paramset)

    return parameterview


def evaluate_parameterset(ps_id, paramset):
    """
    Set paths, derive parameters and write jobscripts for this prameter set.

    Parameters
    ----------
    ps_id
        Unique parameter set id.
    paramset
        Parameter set corresponding to ps_id.    

    """
                
    # set paths and create directories for parameters, jobscripts and
    # raw and processed output data
    for dname in ['parameters', 'jobscripts', 'raw_data', 'processed_data']:
        path = os.path.join(paramset['sim_dict']['data_path'], dname, ps_id)
        if not os.path.isdir(path):
            os.makedirs(path) # also creates sub directories
        paramset['sim_dict']['path_' + dname] = path

    # compute dependent network parameters
    paramset['net_dict'], paramset['stim_dict'] = \
        helpnet.derive_dependent_parameters(
            paramset['net_dict'], paramset['stim_dict'])

    # write final parameters to file (TODO consider human-readable .json)
    for dic in ['sim_dict', 'net_dict', 'stim_dict']:
        with open(os.path.join(paramset['sim_dict']['path_parameters'],
            dic + '.pkl'), 'wb') as f:
            pickle.dump(paramset[dic], f)

    # TODO just for testing
    if 1:
        for key in sorted(paramset['net_dict']):
            print(key)
            print(paramset['net_dict'][key])
            print()

    
    # write jobscripts
    run_cmd = \
        'python3 model_nest/run_mesocircuit.py ' + paramset['sim_dict']['path_parameters']

    if paramset['sim_dict']['computer'] == 'local':
        jobscript = (
            '#!/bin/bash -x' + '\n' +
            'mpirun -n ' + str(paramset['sim_dict']['num_mpi_per_node']) + ' ' +
            run_cmd)
    elif paramset['sim_dict']['computer'] == 'jureca':
        raise Exception # TODO add juerca

    with open(os.path.join(paramset['sim_dict']['path_jobscripts'],
        'network.sh'), 'w') as f:
        f.write(jobscript)

    return

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
        if type(value) == dict or type(value) == ps.ParameterSet:
            y = sorted(iter(list(value.items())), key=operator.itemgetter(0))
            x[i] = (key, y)
            for j, (k, v) in enumerate(y):
                if type(v) == dict or type(v) == ps.ParameterSet:
                    y[j] = (k, sort_deep_dict(v))
    return x
