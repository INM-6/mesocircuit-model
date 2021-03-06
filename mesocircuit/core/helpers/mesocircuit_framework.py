"""PyNEST Mesocircuit: Mesocircuit Framework
--------------------------------------------

Parameterspace evaluation and job execution.

"""

import numpy as np
import parameters as ps
import os
import sys
import subprocess
import shutil
import glob
import operator
import pickle
import json
import yaml
import hashlib
import copy

from ..parameterization import helpers_network as helpnet

# default parameters
from ..parameterization.base_system_params import sys_dict
from ..parameterization.base_simulation_params import sim_dict
from ..parameterization.base_network_params import net_dict
from ..parameterization.base_analysis_params import ana_dict
from ..parameterization.base_plotting_params import plot_dict


def auto_data_directory():
    """
    Automatically determine a data directory.
    """
    try:
        data_dir = os.path.join(os.environ['SCRATCH'],
                                os.environ['USER'],
                                'mesocircuit_data')
    except BaseException:
        data_dir = os.path.join(os.getcwd(), 'data')
    return data_dir


def evaluate_parameterspaces(
        custom_ps_dicts='', paramspace_keys=[], with_base_params=False,
        data_dir=auto_data_directory()):
    """
    Evaluates the parameter spaces as specified by the arguments.

    Parameters
    ----------
    custom_ps_dicts
        Dictonary defining custom parameter spaces to be evaluated.
        If no dictionary is given, i.e., custom_ps_dicts='', parameters are
        evaluated according to the base parameters (default='').
    paramspace_keys
        List of keys of parameter spaces defined in custom_ps_dicts.
        Providing an empty list means that all keys are evaluated (default=[]).
    with_base_params
        Whether to include a parameter space with only base parameters
        (default=False).
    data_dir
        Absolute path to write data to.

    """

    ps_dicts = {}

    if custom_ps_dicts != '':
        ps_dicts.update(custom_ps_dicts)
    if with_base_params:
        ps_dicts.update({'base': {}})

    # parameterspaces built with the parameters module and indexed by
    # paramspace_key
    parameterspaces = {}

    for paramspace_key in sorted(ps_dicts):
        if (len(paramspace_keys) == 0 or  # all keys
            paramspace_key in paramspace_keys or  # selected key(s)
                paramspace_key == 'base'):  # base parameters if with_base_params

            parameterspaces[paramspace_key] = ps.ParameterSpace({})
            # start with default parameters and update
            for dic, vdic in zip(
                ['sys_dict', 'sim_dict', 'net_dict', 'ana_dict', 'plot_dict'],
                    [sys_dict, sim_dict, net_dict, ana_dict, plot_dict]):
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
                    'sys_dict': parameterspaces[paramspace_key]['sys_dict'],
                    'ana_dict': parameterspaces[paramspace_key]['ana_dict'],
                    'plot_dict': parameterspaces[paramspace_key]['plot_dict']}

                # full path to data directory
                full_data_path = os.path.join(data_dir, paramspace_key, ps_id)
                print(f'    Data path: {full_data_path}')

                evaluate_parameterset(ps_id, paramset, full_data_path)
    return


def evaluate_parameterset(ps_id, paramset, full_data_path):
    """
    Set paths, derive parameters and write jobscripts for this parameter set.

    Parameters
    ----------
    ps_id
        Unique parameter set id.
    paramset
        Parameter set corresponding to ps_id.
    full_data_path
        Full path to data directory, including parameterspace and ps_id.
    """

    # set paths and create directories for parameters, jobscripts and
    # raw and processed output data
    for dname in \
        ['code', 'parameters', 'jobscripts', 'raw_data', 'processed_data',
         'plots', 'stdout']:
        path = os.path.join(full_data_path, dname)
        if not os.path.isdir(path):
            os.makedirs(path)  # also creates sub directories

    # compute dependent network parameters
    paramset['net_dict'] = \
        helpnet.derive_dependent_parameters(paramset['net_dict'])

    # write final parameters to file
    for dic in ['sys_dict', 'sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
        filename = os.path.join(full_data_path, 'parameters', dic)
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
    filename = os.path.join(full_data_path, 'parameters', 'lmt_dict')
    with open(filename + '.yaml', 'w') as f:
        yaml.dump(lmt_dic, f, default_flow_style=False)
    shutil.copyfile(os.path.join('core/parameterization',
                                 'lmt_analysis_params.yaml'),
                    os.path.join(full_data_path, 'parameters',
                                 'lmt_ana_dict.yaml'))

    # copy code
    for d in ['simulation', 'analysis', 'plotting', 'helpers']:
        dir_path = os.path.join(full_data_path, 'code', 'core', d)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
    for f in [
        'run_network.py',
        'run_analysis.py',
        'run_plotting.py',
        'core/simulation/network.py',
        'core/analysis/spike_analysis.py',
        'core/stats.py',  # TODO move stats into analysis
        'core/plotting/plotting.py',
        'core/plotting/figures.py',
        'core/helpers/base_class.py',
        'core/helpers/mpiops.py',
        'core/helpers/parallelism_time.py'
    ]:
        shutil.copyfile(f, os.path.join(full_data_path, 'code', f))

    # write jobscripts
    write_jobscripts(paramset['sys_dict'], full_data_path)
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
    if net_dict['delay_type'] == 'normal':
        d_e_mean = net_dict['delay_exc_mean']
        d_i_mean = net_dict['delay_inh_mean']
        d_e_sd = d_e_mean * net_dict['delay_rel_std']
        d_i_sd = d_i_mean * net_dict['delay_rel_std']

    elif net_dict['delay_type'] == 'linear':
        # get columns from exc. or inh. sources and average
        d_e_mean = float(np.mean(net_dict['delay_lin_eff_mean'][:, ::2]))
        d_i_mean = float(np.mean(net_dict['delay_lin_eff_mean'][:, 1::2]))
        d_e_sd = float(np.mean(net_dict['delay_lin_eff_std'][:, ::2]))
        d_i_sd = float(np.mean(net_dict['delay_lin_eff_std'][:, 1::2]))

    dic = {
        # for correct parameter derivations, includes doubled weight L4E->L23E,
        # but rel_weight_exc_to_inh is not covered
        'label': 'microcircuit',
        'populations': net_dict['populations'][:-1].tolist(),  # no thalamus
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
        'd_e': {'val': d_e_mean,
                'unit': 'ms'},
        'd_i': {'val': d_i_mean,
                'unit': 'ms'},
        'd_e_sd': {'val': d_e_sd,
                   'unit': 'ms'},
        'd_i_sd': {'val': d_i_sd,
                   'unit': 'ms'},
        'delay_dist': 'gaussian',  # not exact, but better than none
        # use L23E -> L23E
        'w': {'val': net_dict['full_weight_matrix_mean'][0][0].tolist(),
              'unit': 'pA'},
        'K': net_dict['full_indegrees'][:, :-1].tolist(),
        'g': - net_dict['g'],
        'nu_ext': {'val': net_dict['bg_rate'],
                   'unit': 'Hz'},
        'K_ext': net_dict['full_ext_indegrees'].tolist(),
        'nu_e_ext': {'val': np.zeros(8).tolist(),
                     'unit': 'Hz'},
        'nu_i_ext': {'val': np.zeros(8).tolist(),
                     'unit': 'Hz'}}
    return dic


def write_jobscripts(sys_dict, path):
    """
    Writes a jobscript for each machine (hpc, local) and each step
    (network, analysis, plotting, analyis_and_plotting) specified in the system
    parameters.

    Parameters
    ----------
    sys_dict
        Dictionary with system specifications.
    path
        Path to folder of ps_id.
    """

    for machine, dic in sys_dict['machines'].items():
        machine_path = os.path.join(path, machine)
        for name, scripts in [['network', ['run_network.py']],
                              ['analysis', ['run_analysis.py']],
                              ['plotting', ['run_plotting.py']],
                              ['analysis_and_plotting', ['run_analysis.py',
                                                         'run_plotting.py']]]:

            # key of sys_dict defining resources
            res = name if name == 'network' else 'analysis_and_plotting'
            dic = sys_dict['machines'][machine][res]

            # file for output and errors
            stdout = os.path.join('stdout', name + '.txt')

            # start jobscript
            jobscript = ('#!/bin/bash -x\n')

            # define machine specifics
            if machine == 'hpc':
                # assume SLURM, append resource definitions
                jobscript += (
                    "#SBATCH --job-name=meso\n"
                    f"#SBATCH --partition={dic['partition']}\n"
                    f"#SBATCH --output={stdout}\n"
                    f"#SBATCH --error={stdout}\n"
                    f"#SBATCH --nodes={dic['num_nodes']}\n"
                    f"#SBATCH --ntasks-per-node={dic['num_mpi_per_node']}\n"
                    f"#SBATCH --cpus-per-task={dic['local_num_threads']}\n"
                    f"#SBATCH --time={dic['wall_clock_time']}\n"
                    "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n\n")
                run_cmd = 'srun --mpi=pmi2'

            elif machine == 'local':
                run_cmd = f"mpirun -n {dic['num_mpi']}"

            # append executable(s),
            # number of local threads needed for network simulation,
            # tee output to file for local execution (append for multiple jobs)
            t = dic['local_num_threads'] if name == 'network' else ''
            o_0 = f'2>&1 | tee {stdout}' if machine == 'local' else ''
            o_1 = f'2>&1 | tee -a {stdout}' if machine == 'local' else ''
            executables = [
                f'{run_cmd} python3 -u code/{py} {t} {o_0 if i == 0 else o_1}'
                for i, py in enumerate(scripts)]
            sep = '\n\n' + 'wait' + '\n\n'
            jobscript += sep.join(executables)

            # write jobscript
            fname = os.path.join(path, 'jobscripts', f"{machine}_{name}.sh")
            with open(fname, 'w') as f:
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


def run_jobs(
        paramspace_keys=[], with_base_params=False,
        jobs=['network', 'analysis_and_plotting'],
        machine='hpc',
        data_dir=auto_data_directory()):
    """
    Initiatiates job execution for all given parameter spaces.

    Parameters
    ----------
    paramspace_keys
        List of keys of parameter spaces defined in custom_ps_dicts.
        Providing an empty list means that all keys are evaluated (default=[]).
    with_base_params
        Whether to include a parameter space with only base parameters
        (default=False).
    jobs
        List of one or multiple of 'network, 'analysis, 'plotting', and
        'anlysis_and_plotting'.
    machine
        'local' or 'hpc'.
    data_dir
        Absolute path to write data to.
    """

    ps_keys = paramspace_keys
    if with_base_params:
        ps_keys.append('base')

    # parameters spaces identified by key
    for ps_key in ps_keys:
        full_data_paths = glob.glob(os.path.join(data_dir, ps_key, '*'))
        # parameter sets identified by ps_id
        for full_data_path in full_data_paths:
            ps_id = os.path.basename(full_data_path)
            run_single_jobs(ps_key, ps_id, jobs, machine, data_dir)
    return


def run_single_jobs(paramspace_key, ps_id,
                    jobs=['network', 'analysis_and_plotting'], machine='hpc',
                    data_dir=auto_data_directory()):
    """
    Runs jobs of a single parameterset.

    Parameters
    ----------
    paramspace_key
        A key identifying a parameter space.
    ps_id
        A parameter space id.
    jobs
        List of one or multiple of 'network, 'analysis, 'plotting', and
        'anlysis_and_plotting'.
    job
        'network', 'analysis', 'plotting', or 'analysis_and_plotting'.
    machine
        'local' or 'hpc'.
    data_dir
        Absolute path to write data to.
    """
    # change to directory with copied files
    full_data_path = os.path.join(data_dir, paramspace_key, ps_id)
    os.chdir(full_data_path)

    jobinfo = ' and '.join(jobs) if len(jobs) > 1 else jobs[0]
    info = (f'{jobinfo} for {paramspace_key} - {ps_id}.' + '\n'
            f'    Data path: {full_data_path}')

    if machine == 'hpc':
        print('Submitting ' + info)
        submit = f'sbatch --account $BUDGET_ACCOUNTS jobscripts/{machine}_{jobs[0]}.sh'
        output = subprocess.getoutput(submit)
        print(output)
        jobid = output.split(' ')[-1]
        # submit potential following jobs with dependency
        if len(jobs) > 1:
            for i, job in enumerate(jobs[1:]):
                submit = (
                    f'sbatch --account $BUDGET_ACCOUNTS ' +
                    f'--dependency=afterok:{jobid} jobscripts/{machine}_{job}.sh')
                output = subprocess.getoutput(submit)
                print(output)
                jobid = output.split(' ')[-1]

    elif machine == 'local':
        print('Running ' + info)
        for job in jobs:
            os.system(f'sh jobscripts/{machine}_{job}.sh')
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
