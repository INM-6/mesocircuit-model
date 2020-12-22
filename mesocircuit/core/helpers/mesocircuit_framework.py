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
import json
import yaml
import hashlib
import copy

from ..parameterization import helpers_network as helpnet

# default parameters
from ..parameterization.base_sim_params import sim_dict
from ..parameterization.base_network_params import net_dict
from ..parameterization.base_analysis_params import ana_dict
from ..parameterization.base_plotting_params import plot_dict


def evaluate_parameterspaces(
        custom_ps_dicts='', paramspace_keys=[], with_base_params=False):
    """
    Evaluates the parameter spaces as specified by the arguments.

    Parameters
    ----------
    custom_ps_dicts
        Dictonary defining custom parameter spaces to be evaluated.
        If no dictionary is given, i.e., custom_ps_dicts='', parameters are
        evaluated according to the base parameters (default='').
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

    if custom_ps_dicts != '':
        ps_dicts.update(custom_ps_dicts)
    if with_base_params:
        ps_dicts.update({'base': {}})

    # parameterspaces built with the parameters module and indexed by
    # paramspace_key
    parameterspaces = {}
    # overview of parameterspaces and corresponding ps_ids
    parameterview = {}

    for paramspace_key in sorted(ps_dicts):
        if (len(paramspace_keys) == 0 or  # all keys
            paramspace_key in paramspace_keys or  # selected key(s)
                paramspace_key == 'base'):  # base parameters if with_base_params
            parameterview[paramspace_key] = []

            parameterspaces[paramspace_key] = ps.ParameterSpace({})
            # start with default parameters and update
            for dic, vdic in zip(
                ['sim_dict', 'net_dict', 'ana_dict', 'plot_dict'],
                    [sim_dict, net_dict, ana_dict, plot_dict]):
                parameterspaces[paramspace_key][dic] = dict(
                    vdic)  # copy is needed
                if dic in ps_dicts[paramspace_key]:
                    parameterspaces[paramspace_key][dic].update(
                        ps_dicts[paramspace_key][dic])

            # only sim_dict and net_dict enable parameter spaces and are used to
            # compute a unique id
            dicts_unique = ['sim_dict', 'net_dict']
            sub_paramspace = ps.ParameterSpace(
                {k: parameterspaces[paramspace_key][k] for k in dicts_unique})

            for sub_paramset in sub_paramspace.iter_inner():
                ps_id = get_unique_id(sub_paramset)
                print('Evaluating parameters for ' + str(paramspace_key) +
                      ' - ' + str(ps_id) + '.')

                # readd ana_dict and plot_dict to get full paramset
                # (deep copy of sub_paramset is needed, otherwise changes to
                # paramset['sim_dict']['data_path'] survive iterations)
                paramset = {
                    **copy.deepcopy(sub_paramset),
                    'ana_dict': parameterspaces[paramspace_key]['ana_dict'],
                    'plot_dict': parameterspaces[paramspace_key]['plot_dict']}

                # add paramspace_key to data_path
                paramset['sim_dict']['data_path'] = os.path.join(
                    paramset['sim_dict']['data_path'], paramspace_key)

                evaluate_parameterset(ps_id, paramset)

                parameterview[paramspace_key].append([
                    paramset['sim_dict']['data_path'], ps_id])

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
            os.makedirs(path)  # also creates sub directories
        paramset['sim_dict']['path_' + dname] = path

    # compute dependent network parameters
    paramset['net_dict'] = \
        helpnet.derive_dependent_parameters(paramset['net_dict'])

    # write final parameters to file
    for dic in ['sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
        filename = os.path.join(paramset['sim_dict']['path_parameters'], dic)
        # pickle for machine readability
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(paramset[dic], f)
        # text for human readability
        with open(filename + '.txt', 'w') as f:
            json_dump = json.dumps(
                paramset[dic], cls=NumpyEncoder, indent=2, sort_keys=True)
            f.write(json_dump)
    # parameters for LIF Meanfield Tools
    lmt_dic = params_for_lif_meanfield_tools(paramset['net_dict'])
    filename = os.path.join(paramset['sim_dict']['path_parameters'], 'lmt_dict')
    with open(filename + '.yaml', 'w') as f:
        yaml.dump(lmt_dic, f, default_flow_style=False)

    # write jobscripts
    write_jobscript('network.sh', paramset)
    write_jobscript('analysis.sh', paramset)
    write_jobscript('lfp.sh', paramset)
    write_jobscript('plotting.sh', paramset)
    write_jobscript('analysis_and_plotting.sh', paramset)
    return


def params_for_lif_meanfield_tools(net_dict):
    """
    Creates a dictionary with parameters for mean-field theoretical analysis
    with LIF Meanfield Tools (https://github.com/INM-6/lif_meanfield_tools).

    The parameters for the full network are used.
    Currently the normal delay values are taken independent of which delay type
    is chosen.

    Parameters
    ----------
    net_dict
        Final network dictionary.
    """
    dic = {
        'label': 'microcircuit', # for correct parameter derivations
        'populations': net_dict['populations'][:-1].tolist(), # no thalamus
        'N': net_dict['full_num_neurons'][:-1].tolist(),
        'C': {'val': net_dict['neuron_params']['C_m'],
              'unit': 'pF'},
        'tau_m': {'val': net_dict['neuron_params']['tau_m'],
                  'unit': 'ms'},
        'tau_r': {'val': net_dict['neuron_params']['t_ref'],
                  'unit': 'ms'},
        'V_0_abs': {'val': net_dict['neuron_params']['V_reset'],
                    'unit': 'mV'},
        'V_th_abs': {'val': net_dict['neuron_params']['V_th'],
                     'unit': 'mV'},
        'tau_s': {'val': net_dict['neuron_params']['tau_syn'],
                  'unit': 'ms'},
        # TODO currently only the values from normally distributed delays are
        # taken
        'd_e': {'val': net_dict['delay_exc_mean'],
                'unit': 'ms'},
        'd_i': {'val': net_dict['delay_inh_mean'],
                'unit': 'ms'},
        'd_e_sd': {'val': net_dict['delay_exc_mean'] * net_dict['delay_rel_std'],
                   'unit': 'ms'}, 
        'd_i_sd': {'val': net_dict['delay_inh_mean'] * net_dict['delay_rel_std'],
                   'unit': 'ms'},
        'delay_dist': 'none',
        # use L23E -> L23E
        'w': {'val': net_dict['full_weight_matrix_mean'][0][0].tolist(),
              'unit': 'pA'},
        'K': net_dict['full_indegrees'][:,:-1].tolist(),
        'g': - net_dict['g'],
        'nu_ext': {'val': net_dict['bg_rate'],
                   'unit': 'Hz'},
        'K_ext': net_dict['full_ext_indegrees'].tolist(),
        'nu_e_ext': {'val': np.zeros(8).tolist(),
                     'unit': 'Hz'},
        'nu_i_ext': {'val': np.zeros(8).tolist(),
                     'unit': 'Hz'}}
    return dic


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
        run_py = ['run_network.py']
        dic = paramset['sim_dict']
    elif jsname == 'analysis.sh':
        run_py = ['run_analysis.py']
        dic = paramset['ana_dict']
    elif jsname == 'lfp.sh':
        run_py = ['run_lfp_simulation.py']
        dic = paramset['sim_dict']
    elif jsname == 'plotting.sh':
        run_py = ['run_plotting.py']
        dic = paramset['plot_dict']
    elif jsname == 'analysis_and_plotting.sh':
        run_py = ['run_analysis.py', 'run_plotting.py']
        dic = paramset['ana_dict']  # use configuration from analysis

    # computer-dependent run command
    if dic['computer'] == 'local':
        run_cmd = 'mpirun '
    elif dic['computer'] == 'jureca':
        run_cmd = 'srun '

    # define executable
    executable = [run_cmd + 'python3 -u ' + os.path.join(os.getcwd(), py) + ' ' +
                  paramset['sim_dict']['path_parameters'] for py in run_py]
    sep = '\n\n' + 'wait' + '\n\n'
    executable = sep.join(executable)

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
        if isinstance(value, dict) or isinstance(value, ps.ParameterSet):
            y = sorted(iter(list(value.items())), key=operator.itemgetter(0))
            x[i] = (key, y)
            for j, (k, v) in enumerate(y):
                if isinstance(v, dict) or isinstance(v, ps.ParameterSet):
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
                    jobs = [
                        os.path.join(
                            data_path,
                            'jobscripts',
                            ps_id,
                            js) for js in jobscripts]

                    job_spec = ' for ' + paramspace_key + ' - ' + ps_id + '.'

                    # run locally one job after the other
                    if run_type == 'run_locally':
                        for i, js in enumerate(jobs):
                            print('Running ' + jobscripts[i] + job_spec)
                            os.system('sh ' + js)

                    # submit jobs to jureca
                    elif run_type == 'submit_jureca':
                        # submit first job
                        print('Submitting ' + jobscripts[0] + job_spec)
                        submit = 'sbatch ' + jobs[0]
                        output = subprocess.getoutput(submit)
                        print(output)
                        jobid = output.split(' ')[-1]
                        # submit potential following jobs with dependency
                        if len(jobs) > 1:
                            for i, js in enumerate(jobs[1:]):
                                print('Submitting ' +
                                      jobscripts[i + 1] + job_spec)
                                submit = 'sbatch --dependency=afterok:' + jobid + ' ' + js
                                output = subprocess.getoutput(submit)
                                print(output)
                                jobid = output.split(' ')[-1]
                    submitted_jobs.append(ps_id)
    return


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
