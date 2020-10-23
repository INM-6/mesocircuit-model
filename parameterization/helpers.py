"""PyNEST Mesocircuit: Helper Functions
---------------------------------------

General helper functions.

"""

import numpy as np
import parameters as ps
import os
import sys
import subprocess
import operator
import pickle
import hashlib

from . import helpers_network_stimulus as helpnet

# default parameters
from .base_sim_params import sim_dict
from .base_network_params import net_dict
from .base_stimulus_params import stim_dict
from .base_analysis_params import ana_dict
from .base_plotting_params import plot_dict


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
        if filename != '':
            print('  Check for syntax error in ' + filename + '.py.') 
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
            parameterview[paramspace_key] = []

            parameterspaces[paramspace_key] = ps.ParameterSpace({})
            # start with default parameters and update
            for dic,vdic in zip(
                ['sim_dict', 'net_dict', 'stim_dict', 'ana_dict', 'plot_dict'],
                [sim_dict, net_dict, stim_dict, ana_dict, plot_dict]):
                parameterspaces[paramspace_key][dic] = dict(vdic) # copy is needed
                if dic in ps_dicts[paramspace_key]:
                    parameterspaces[paramspace_key][dic].update(
                    ps_dicts[paramspace_key][dic])

            for paramset in parameterspaces[paramspace_key].iter_inner():
                # include only sim_dict, net_dict and stim_dict into unique id
                ps_id = get_unique_id(
                    {key: paramset[key] for key in \
                        ['sim_dict', 'net_dict', 'stim_dict']})
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
    for dname in \
        ['parameters', 'jobscripts', 'raw_data', 'processed_data', 'plots',
         'stdout']:
        path = os.path.join(paramset['sim_dict']['data_path'], dname, ps_id)
        if not os.path.isdir(path):
            os.makedirs(path) # also creates sub directories
        paramset['sim_dict']['path_' + dname] = path

    # compute dependent network parameters
    paramset['net_dict'], paramset['stim_dict'] = \
        helpnet.derive_dependent_parameters(
            paramset['net_dict'], paramset['stim_dict'])

    # write final parameters to file (TODO consider human-readable .json)
    for dic in ['sim_dict', 'net_dict', 'stim_dict', 'ana_dict', 'plot_dict']:
        with open(os.path.join(paramset['sim_dict']['path_parameters'],
            dic + '.pkl'), 'wb') as f:
            pickle.dump(paramset[dic], f)

    # TODO just for testing
    if 0:
        for key in sorted(paramset['net_dict']):
            print(key)
            print(paramset['net_dict'][key])
            print()

    
    # write jobscripts
    write_jobscript('network.sh', paramset)
    write_jobscript('analysis.sh', paramset)
    write_jobscript('plotting.sh', paramset)
    write_jobscript('analysis_and_plotting.sh', paramset)
    return


def write_jobscript(jsname, paramset):
    """
    Writes a jobscript for the given parameter set.

    Parameters
    ----------
    jsname
        String defining the jobscripts. Options are 'network.sh' and
        'analysis.sh'.
    paramset
        A parameter set.
    """
    
    if jsname == 'network.sh':
        run_py = ['model_nest/run_mesocircuit.py']
        dic = paramset['sim_dict']
    elif jsname == 'analysis.sh':
        run_py = ['analysis/run_analysis.py']
        dic = paramset['ana_dict']
    elif jsname == 'plotting.sh':
        run_py = ['plotting/run_plotting.py']
        dic = paramset['plot_dict']
    elif jsname == 'analysis_and_plotting.sh':
        run_py = ['analysis/run_analysis.py', 'plotting/run_plotting.py']
        dic = paramset['ana_dict'] # use configuration from analysis

    # computer-dependent run command
    if dic['computer'] == 'local':
        # use mpirun only for more than 1 MPI processes
        if dic['num_mpi_per_node'] > 1:
            run_cmd = 'mpirun -n {} '.format(dic['num_mpi_per_node'])
        else:
            run_cmd = ''
    elif dic['computer'] == 'jureca':
        run_cmd = 'srun '

    # define executable
    executable = [run_cmd + 'python3 ' + os.path.join(os.getcwd(), py) + ' ' +
                  paramset['sim_dict']['path_parameters'] for py in run_py]
    executable = '\n\n wait \n\n'.join(executable)

    # start jobscript
    jobscript = ('#!/bin/bash -x' + '\n')

    if dic['computer'] == 'local':
        jobscript += executable

    elif dic['computer'] == 'jureca':
        cores = 24
        # local_num_threads is only defined in sim_dict
        if 'local_num_threads' not in dic:
            threads = int(cores / dic['num_mpi_per_node'])
        else:
            threads = dic['local_num_threads']
        
        stdout = os.path.join(paramset['sim_dict']['path_stdout'],
                              jsname.split('.')[0] + '.txt')

        sbatch = (
            '#SBATCH --account=jinb33\n' +
            '#SBATCH --nodes={}\n'.format(dic['num_nodes']) +
            '#SBATCH --ntasks-per-node={}\n'.format(dic['num_mpi_per_node']) +
            '#SBATCH --cpus-per-task={}\n'.format(threads) +
            '#SBATCH --output={}\n'.format(stdout) +
            '#SBATCH --error={}\n'.format(stdout) +
            '#SBATCH --time={}\n'.format(dic['wallclock_time']) +
            '#SBATCH --partition=batch\n\n' +
            'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}\n\n')

        jobscript += sbatch
        jobscript += executable

    with open(os.path.join(paramset['sim_dict']['path_jobscripts'],
        jsname), 'w') as f:
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


def run_jobs(parameterview, jobscripts, run_type='run_locally',
             paramspace_keys=[]):
    """
    Submits given jobscripts of all parameter combinations in parameterview to
    JURECA.
    
    Parameters
    ----------
    parameterview
        returned by evaluate_parameterspaces()
    jobscripts
        list of jobscripts to be submitted,
        e.g., ['network.sh', 'analysis_plotting.sh'].
        If multiple scripts are given, they will be executed in the given order,
        (on JURECA combined as job arrays).
    run_type
        'run_locally' executes all jobscripts one after the other.
        'submit_jureca' submits all jobscripts to jureca.
    paramspace_keys
        List of keys of parameter spaces to run jobs of. Providing an
        empty list means that all keys are evaluated (default=[])
    """
    submitted_jobs = []
    for paramspace_key in parameterview.keys():
        if paramspace_keys != [] and paramspace_key not in paramspace_keys:
            pass
        else:
            for data_path, ps_id in parameterview[paramspace_key]:
                if ps_id in submitted_jobs:
                    pass
                else:
                    jobs = [os.path.join(
                        data_path, 'jobscripts', ps_id, js) for js in jobscripts]

                    job_spec = ' for ' + paramspace_key + ' - ' + ps_id + '.' 

                    # run locally one job after the other
                    if run_type == 'run_locally':
                        for i,js in enumerate(jobs):
                            print('Running ' + jobscripts[i] + job_spec)
                            os.system('sh ' +  js)

                    # submit jobs to jureca
                    elif run_type == 'submit_jureca':
                        # submit first job
                        print('Submitting ' + jobscripts[1] + job_spec)
                        jobid = subprocess.getoutput('sbatch {}'.format(jobs[0]))
                        # submit potential following jobs with dependency
                        if len(jobs) > 1:
                            for i,js in enumerate(jobs[1:]):
                                print('Submitting ' + jobscripts[i+1] + job_spec)
                                jobid = subprocess.getoutput(
                                    'sbatch --dependency=afterok:{} {}'.format(
                                        jobid, js))
                    submitted_jobs.append(ps_id)
    return