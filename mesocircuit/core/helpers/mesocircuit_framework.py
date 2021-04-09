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


def extend_existing_parameterspaces(
        custom_key, custom_params, base_key, base_ps_dicts):
    """
    Adds a new parameter space to existing dictionary.

    Parameters
    ----------
    custom_key
        New parameter space key.
    custom_params
        New parameters (may include ranges).
    base_key
        New dictionary with custom_key will be based on this dictionary.
    base_ps_dicts
        Base parameter space dictionaries containing a parameter space (or set)
        corresponding to base_key.
    """
    custom_ps_dicts = dict(base_ps_dicts)
    custom_ps_dicts[custom_key] = dict(custom_ps_dicts[base_key])
    for dic in custom_params.keys():
        if dic in custom_ps_dicts[custom_key].keys():
            custom_ps_dicts[custom_key][dic].update(custom_params[dic])
        else:
            custom_ps_dicts[custom_key][dic] = custom_params[dic]

    return custom_ps_dicts


def evaluate_parameterspaces(
        custom_ps_dicts='', paramspace_keys=[], with_base_params=False,
        data_dir=auto_data_directory()):
    """
    Evaluates the parameter spaces as specified by the arguments.

    Parameters
    ----------
    custom_ps_dicts
        Dictionary defining custom parameter spaces to be evaluated.
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

    Returns
    -------
    parameterview
        Dictionary of evaluated parameter spaces as overview.
    """
    ps_dicts = {}

    if custom_ps_dicts != '':
        ps_dicts.update(custom_ps_dicts)
    if with_base_params:
        ps_dicts.update({'base': {}})

    print(f'Data directory: {data_dir}')

    # parameterspaces built with the parameters module and indexed by
    # paramspace_key
    parameterspaces = {}
    # overview of parameterspaces and corresponding ps_ids
    parameterview = {}

    for paramspace_key in sorted(ps_dicts):
        if (len(paramspace_keys) == 0 or  # all keys
            paramspace_key in paramspace_keys or  # selected key(s)
                (with_base_params and paramspace_key == 'base')):
            parameterview[paramspace_key] = {}
            parameterview[paramspace_key]['custom_params'] = {}
            parameterview[paramspace_key]['custom_params']['ranges'] = {}
            parameterview[paramspace_key]['custom_params']['values'] = {}

            parameterspaces[paramspace_key] = ps.ParameterSpace({})
            # start with default parameters and update
            for dic, vdic in zip(
                ['sys_dict', 'sim_dict', 'net_dict', 'ana_dict', 'plot_dict'],
                    [sys_dict, sim_dict, net_dict, ana_dict, plot_dict]):
                parameterspaces[paramspace_key][dic] = dict(
                    vdic)  # copy is needed
                if dic in ps_dicts[paramspace_key]:
                    parameterspaces[paramspace_key][dic] = __merge_dictionaries(
                        parameterspaces[paramspace_key][dic],
                        ps_dicts[paramspace_key][dic])

                    # ranges and values from parameter space that overwrite the
                    # base parameters
                    # TODO: this is not implemented for deeper dictionaries, yet
                    # (could be handled as __merge_dictionaries())
                    for param, val in ps_dicts[paramspace_key][dic].items():
                        if isinstance(val, ps.ParameterRange):
                            if dic not in parameterview[paramspace_key][
                                    'custom_params']['ranges'].keys():
                                parameterview[paramspace_key][
                                    'custom_params']['ranges'][dic] = {}
                            parameterview[paramspace_key][
                                'custom_params']['ranges'][dic][param] = list(val)
                        else:
                            if dic not in parameterview[paramspace_key][
                                    'custom_params']['values'].keys():
                                parameterview[paramspace_key][
                                    'custom_params']['values'][dic] = {}
                            parameterview[paramspace_key][
                                'custom_params']['values'][dic][param] = val

            # only sim_dict and net_dict enable parameter spaces and are used to
            # compute a unique id
            dicts_unique = ['sim_dict', 'net_dict']
            sub_paramspace = ps.ParameterSpace(
                {k: parameterspaces[paramspace_key][k] for k in dicts_unique})

            parameterview[paramspace_key]['paramsets'] = {}
            for sub_paramset in sub_paramspace.iter_inner():
                ps_id = get_unique_id(sub_paramset)
                print(f'Evaluating parameters for {paramspace_key} - {ps_id}.')

                # readd ana_dict and plot_dict to get full paramset
                # (deep copy of sub_paramset is needed)
                paramset = {
                    **copy.deepcopy(sub_paramset),
                    'sys_dict': parameterspaces[paramspace_key]['sys_dict'],
                    'ana_dict': parameterspaces[paramspace_key]['ana_dict'],
                    'plot_dict': parameterspaces[paramspace_key]['plot_dict']}

                # add parameterset values of ranges to parameterview
                parameterview[paramspace_key]['paramsets'][ps_id] = {}
                for dic in \
                        parameterview[paramspace_key]['custom_params']['ranges']:
                    parameterview[paramspace_key][
                        'paramsets'][ps_id][dic] = {}
                    for param, val in \
                        parameterview[paramspace_key][
                            'custom_params']['ranges'][dic].items():
                        parameterview[paramspace_key][
                            'paramsets'][ps_id][dic][param] = \
                            paramset[dic][param]

                # evaluate the parameter set
                evaluate_parameterset(
                    ps_id, paramset,
                    os.path.join(data_dir, paramspace_key, ps_id))

            # setup for parameterspace analysis
            for dname in ['parameters', 'plots']:
                path = os.path.join(
                    data_dir, paramspace_key, 'parameter_space', dname)
                if not os.path.isdir(path):
                    os.makedirs(path)

            # write parameterview to file
            dir = os.path.join(data_dir, paramspace_key,
                               'parameter_space', 'parameters')
            # pickle for machine readability
            with open(os.path.join(dir, 'psview_dict.pkl'), 'wb') as f:
                pickle.dump(parameterview, f)
            # text for human readability
            with open(os.path.join(dir, 'psview_dict.txt'), 'w') as f:
                json_dump = json.dumps(
                    parameterview, cls=NumpyEncoder, indent=2, sort_keys=True)
                f.write(json_dump)

            # sorted list of ranges (if any exist)
            psview_ranges = \
                parameterview[paramspace_key]['custom_params']['ranges']
            ranges = []
            for dic in sorted(psview_ranges.keys()):
                for r in sorted(psview_ranges[dic].keys()):
                    ranges.append([dic, r, psview_ranges[dic][r]])
            dim = len(ranges)  # dimension of parameter space
            if dim not in [1, 2]:
                print(
                    f'Parameterspace {paramspace_key} has dimension {dim}. ' +
                    'Hashes are not printed.')
            else:
                # set up a hash map
                shape = [len(r[2]) for r in ranges]
                hashmap = np.zeros(shape, dtype=object)
                psets = parameterview[paramspace_key]['paramsets']
                for p, h in enumerate(psets.keys()):
                    d0_dict, d0_param, d0_range = ranges[0]
                    for i, val0 in enumerate(d0_range):
                        if psets[h][d0_dict][d0_param] == val0:
                            if dim == 1:
                                hashmap[i] = h
                            else:
                                d1_dict, d1_param, d1_range = ranges[1]
                                for j, val1 in enumerate(d1_range):
                                    if psets[h][d1_dict][d1_param] == val1:
                                        if dim == 2:
                                            hashmap[i][j] = h
                                        else:
                                            d2_dict, d2_param, d2_range = ranges[2]
                if dim == 1:
                    hashmap = hashmap.reshape(-1, 1)

                # write ranges and hashmap to file
                ranges_hashmap = {'ranges': ranges, 'hashmap': hashmap}
                dir = os.path.join(data_dir, paramspace_key,
                                   'parameter_space', 'parameters')
                # pickle for machine readability
                with open(os.path.join(dir, 'ranges_hashmap.pkl'), 'wb') as f:
                    pickle.dump(ranges_hashmap, f)
                # text for human readability
                with open(os.path.join(dir, 'ranges_hashmap.txt'), 'w') as f:
                    for i, r in enumerate(ranges):
                        lst = '[' + ', '.join([str(x) for x in r[2]]) + ']'
                        line = f'dim{i}: {r[0]}[{r[1]}]: {lst}\n'
                        f.write(line)
                    f.write('\n\n')
                    for line in np.matrix(hashmap):
                        np.savetxt(f, line, fmt='%s')

    return parameterview


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
        fname = os.path.join(full_data_path, 'parameters', dic)
        # pickle for machine readability
        with open(fname + '.pkl', 'wb') as f:
            pickle.dump(paramset[dic], f)
        # text for human readability
        with open(fname + '.txt', 'w') as f:
            json_dump = json.dumps(
                paramset[dic], cls=NumpyEncoder, indent=2, sort_keys=True)
            f.write(json_dump)
    # parameters for LIF Meanfield Tools
    lmt_dic = params_for_lif_meanfield_tools(paramset['net_dict'])
    fname = os.path.join(full_data_path, 'parameters', 'lmt_dict')
    with open(fname + '.yaml', 'w') as f:
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
        'run_lfp_simulation.py',
        'run_lfp_plotting.py',
        # 'core/simulation/network.py',
        # 'core/analysis/spike_analysis.py',
        # 'core/analysis/stats.py',
        # 'core/plotting/plotting.py',
        # 'core/plotting/figures.py',
        # 'core/helpers/base_class.py',
        # 'core/helpers/mpiops.py',
        # 'core/helpers/parallelism_time.py',
        # 'core/helpers/io.py'
    ]:
        shutil.copyfile(f, os.path.join(full_data_path, 'code', f))

    # copy 'core' module
    shutil.copytree('core', os.path.join(full_data_path, 'code', 'core'),
                    dirs_exist_ok=True)


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

    for machine, dic in sys_dict.items():
        for name, scripts in [['network', ['run_network.py']],
                              ['analysis', ['run_analysis.py']],
                              ['plotting', ['run_plotting.py']],
                              ['analysis_and_plotting', ['run_analysis.py',
                                                         'run_plotting.py']],
                              ['lfp_simulation', ['run_lfp_simulation.py']],
                              ['lfp_plotting', ['run_lfp_plotting.py']]
                              ]:

            # key of sys_dict defining resources
            res = name if name == 'network' else 'analysis_and_plotting'
            dic = sys_dict[machine][res]

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
                # check which executables are available
                for mpiexec in ['srun', 'mpiexec', 'mpirun']:
                    out = subprocess.run(['which', mpiexec])
                    if out.returncode == 0:
                        if mpiexec in ['mpiexec', 'mpirun']:
                            run_cmd = f'{mpiexec} -n {dic["num_mpi"]}'
                        else:
                            run_cmd = f'{mpiexec} --mpi=pmi2'
                        break

            else:
                raise NotImplementedError(f'machine {machine} not recognized')

            # append executable(s),
            # number of local threads needed for network simulation,
            # tee output to file for local execution (append for multiple jobs)
            t = dic['local_num_threads'] if name == 'network' else ''
            o_0 = f'2>&1 | tee {stdout}' if machine == 'local' else ''
            o_1 = f'2>&1 | tee -a {stdout}' if machine == 'local' else ''
            if name == 'lfp_plotting':
                # should be run serially!
                executables = [
                    f'python3 -u code/{py} {t} {o_0 if i == 0 else o_1}'
                    for i, py in enumerate(scripts)]
            else:
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


def run_parametersets(
        func,
        parameterview=None,
        paramspace_keys=None,
        with_base_params=False,
        ps_ids=[],
        data_dir=auto_data_directory(),
        **kwargs):
    """
    Runs the given function for parameter sets specified.
    Provide either a parameterview or a list of paramspace_keys.

    Parameters
    ----------
    func
        Function to be executed for parameter sets.
    parameterview
        Dictionary of evaluated parameter spaces.
    paramspace_keys
        List of keys of parameter spaces evaluated with
        evaluate_parameterspaces() into data_dir.
    with_base_params
        If paramspace_keys are given:
        Whether to include a parameter space with only base parameters
        (default=False).
    ps_ids
        List of parameterset identifiers (hashes) as computed in
        evaluate_parameterspaces().
        Providing an empty list means that jobs of all ps_ids existing in
        data_dir of the given paramspace_keys are executed (default=[]).
    data_dir
        Absolute path to write data to.
    """
    # note that this comparison is not exhaustive
    boolean = ((parameterview is None and paramspace_keys is None) or
               (parameterview is not None and paramspace_keys is not None))
    if boolean:
        raise Exception('Specify either parameterview or paramspace_keys')

    print(f'Data directory: {data_dir}')

    if parameterview:
        for ps_key in parameterview.keys():
            for ps_id in parameterview[ps_key]['paramsets'].keys():
                if ps_id in ps_ids or ps_ids == []:
                    func(ps_key, ps_id, data_dir, **kwargs)

    elif paramspace_keys:
        ps_keys = paramspace_keys
        if with_base_params:
            ps_keys.append('base')

        for ps_key in ps_keys:
            full_data_paths = glob.glob(os.path.join(data_dir, ps_key, '*'))
            # parameter sets identified by ps_id
            for full_data_path in full_data_paths:
                ps_id = os.path.basename(full_data_path)
                # pass if not a real ps_id (hash)
                if ps_id == 'parameter_space':
                    pass
                if ps_id in ps_ids or ps_ids == []:
                    func(ps_key, ps_id, data_dir, **kwargs)
    return


def run_single_jobs(paramspace_key, ps_id, data_dir=auto_data_directory(),
                    jobs=['network', 'analysis_and_plotting'], machine='hpc'):
    """
    Runs jobs of a single parameterset.

    Parameters
    ----------
    paramspace_key
        A key identifying a parameter space.
    ps_id
        A parameter space id.
    data_diri
        Absolute path to write data to.
    jobs
        List of one or multiple of 'network, 'analysis, 'plotting', and
        'anlysis_and_plotting'.
    job
        'network', 'analysis', 'plotting', or 'analysis_and_plotting'.
    machine
        'local' or 'hpc'.
    """
    # change to directory with copied files
    full_data_path = os.path.join(data_dir, paramspace_key, ps_id)
    cwd = os.getcwd()
    os.chdir(full_data_path)

    jobinfo = ' and '.join(jobs) if len(jobs) > 1 else jobs[0]
    info = f'{jobinfo} for {paramspace_key} - {ps_id}.'

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

    os.chdir(cwd)
    return


def run_single_lmt(paramspace_key, ps_id, data_dir=auto_data_directory()):
    """
    Computes some theoretical quantities with LIF Meanfield Tools for a single
    parameter set.

    TODO move to a more appropriate place

    Parameters
    ----------
    paramspace_key
        A key identifying a parameter space.
    ps_id
        A parameter space id.
    data_dir
        Absolute path to write data to.
    """
    from ..plotting import figures, plotting
    import lif_meanfield_tools as lmt
    ureg = lmt.ureg

    print(f'Computing theory for {paramspace_key} - {ps_id}.')

    os.chdir(os.path.join(data_dir, paramspace_key, ps_id))

    # lmt network object
    nw = lmt.Network(
        network_params=os.path.join(
            'parameters', 'lmt_dict.yaml'), analysis_params=os.path.join(
            'parameters', 'lmt_ana_dict.yaml'))

    # working point
    wp = nw.working_point()

    # power spectrum
    power = nw.power_spectra()
    freqs = nw.analysis_params['omegas'] / (2. * np.pi)

    # sensitivity measure
    pop_idx, freq_idx = np.unravel_index(np.argmax(power),
                                         np.shape(power))
    frequency = freqs[freq_idx]

    sm = nw.sensitivity_measure(freq=frequency)
    eigs = nw.eigenvalue_spectra('MH')
    eigc = eigs[pop_idx][np.argmin(abs(eigs[pop_idx] - 1))]

    Z = nw.sensitivity_measure(frequency)
    k = np.asarray([1, 0]) - np.asarray([eigc.real, eigc.imag])
    k /= np.sqrt(np.dot(k, k))
    k_per = np.asarray([-k[1], k[0]])
    k_per /= np.sqrt(np.dot(k_per, k_per))
    Z_amp = Z.real * k[0] + Z.imag * k[1]
    Z_freq = Z.real * k_per[0] + Z.imag * k_per[1]

    # corresponding plotting class
    dics = []
    for dic in ['sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
        with open(f'parameters/{dic}.pkl', 'rb') as f:
            dics.append(pickle.load(f))
    sim_dict, net_dict, ana_dict, plot_dict = dics

    pl = plotting.Plotting(
        sim_dict, net_dict, ana_dict, plot_dict)

    # overview figure
    figures.theory_overview(
        plot=pl,
        working_point=wp,
        frequencies=freqs,
        power=power,
        sensitvity_amplitude=Z_amp,
        sensitivity_frequency=Z_freq,
        sensitivity_popidx_freq=[pop_idx, frequency])

    return


def __merge_dictionaries(main_dict, new_dict):
    """
    Merges new dictionary recursively into main dictionary.

    Parameters
    ----------
    main_dict
        Main dictionary.
    new_dict
        New dictionary with entries to overwrite main_dict.

    Returns
    -------
    main_dict
        Updated dictionary.
    """
    for key, val in new_dict.items():
        if isinstance(val, dict):
            node = main_dict.setdefault(key, {})
            __merge_dictionaries(node, val)
        else:
            main_dict[key] = val
    return main_dict

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
