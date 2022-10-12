"""PyNEST Mesocircuit: Mesocircuit Framework
--------------------------------------------

Parameterspace evaluation and job execution.

"""

from mesocircuit.parameterization import helpers_network as helpnet
import os
import subprocess
import glob
import pickle
import json
import copy
from time import strftime, gmtime
import yaml
import numpy as np
import parameters as ps

from mesocircuit.helpers import helpers
import mesocircuit.parameterization.base_system_params as base_system_params
import mesocircuit.parameterization.base_simulation_params as base_simulation_params
import mesocircuit.parameterization.base_network_params as base_network_params
import mesocircuit.parameterization.base_analysis_params as base_analysis_params
import mesocircuit.parameterization.base_plotting_params as base_plotting_params

import mesocircuit.lfp.lfp_parameters as lfp_parameters


class MesocircuitExperiment():
    """
    Main mesocircuit class for steering experiments.

    Parameters
    ----------
    name : str
        Name of the experiment. All corresponding data and scripts will be
        written to a folder with this name.
    custom_params : dict, optional
        Dictionary with new parameters or parameter ranges to overwrite the
        default ones.
    data_dir : str, optional
        Absolute path to write data to.
    """

    def __init__(self, name='base', custom_params=None, data_dir=None):
        """
        """
        self.name = name
        print(f'Instantiating MesocircuitExperiment: {name}')

        # data directory
        if not data_dir:
            self.data_dir = self._auto_data_directory()
        else:
            self.data_dir = data_dir
        print(f'Data directory: {self.data_dir}')

        self._evaluate_parameters(custom_params)

    def _auto_data_directory(self, dirname='mesocircuit_data'):
        """
        Automatically determine a data directory.
        """
        try:
            data_dir = os.path.join(os.environ['SCRATCH'],
                                    os.environ['USER'],
                                    dirname)
        except BaseException:
            data_dir = os.path.join(os.getcwd(), dirname)
        return data_dir

    def _evaluate_parameters(self, custom_params):
        """
        Evaluates parameters and creates a parameter view for the experiment.

        Parameters
        ----------
        custom_params : dict, optional
            Dictionary with new parameters or parameter ranges to overwrite the
            default ones.
        """
        # parameterspaces built with the parameters module
        parameterspaces = ps.ParameterSpace({})
        # overview of parameterspaces and corresponding ps_ids
        self.parameterview = {}
        self.parameterview['custom_params'] = {}
        self.parameterview['custom_params']['ranges'] = {}
        self.parameterview['custom_params']['values'] = {}

        # start with default parameters and update
        for dic, vdic in zip(
            ['sys_dict', 'sim_dict', 'net_dict', 'ana_dict', 'plot_dict'],
            [base_system_params.sys_dict,
             base_simulation_params.sim_dict,
             base_network_params.net_dict,
             base_analysis_params.ana_dict,
             base_plotting_params.plot_dict]):

            parameterspaces[dic] = dict(vdic)
            if dic in custom_params:
                parameterspaces[dic] = helpers.merge_dictionaries(
                    parameterspaces[dic], custom_params[dic])

                # insert custom ranges and values into parameterview
                for param, value in custom_params[dic].items():
                    self.parameterview['custom_params'] = \
                        self._custom_params_for_parameterview(
                            dic, param, value)

        # only sim_dict and net_dict are used to compute a unique id
        dicts_unique = ['sim_dict', 'net_dict']
        sub_paramspace = ps.ParameterSpace(
            {k: parameterspaces[k] for k in dicts_unique})

        self.parameterview['paramsets'] = {}
        for sub_paramset in sub_paramspace.iter_inner():
            ps_id = helpers.get_unique_id(sub_paramset)
            print(f'Evaluating parameter set with ID: {ps_id}')

            # readd ana_dict and plot_dict to get full paramset
            # (deep copy of sub_paramset is needed)
            paramset = {
                **copy.deepcopy(sub_paramset),
                'sys_dict': parameterspaces['sys_dict'],
                'ana_dict': parameterspaces['ana_dict'],
                'plot_dict': parameterspaces['plot_dict']}

            # add parameterset values of ranges to parameterview
            self.parameterview['paramsets'][ps_id] = {}
            for dic in self.parameterview['custom_params']['ranges']:
                self.parameterview['paramsets'][ps_id][dic] = {}
                for param, val in \
                        self.parameterview['custom_params']['ranges'][dic].items():
                    self.parameterview[
                        'paramsets'][ps_id][dic][param] = paramset[dic][param]

            # evaluate the parameter set
            self._evaluate_single_parameterset(ps_id, paramset)

        # setup for parameterspace analysis
        for dname in ['parameters', 'plots']:
            path = os.path.join(
                self.data_dir, self.name, 'parameter_space', dname)
            if not os.path.isdir(path):
                os.makedirs(path)

        # write parameterview to file
        dir = os.path.join(self.data_dir, self.name,
                           'parameter_space', 'parameters')
        # pickle for machine readability
        with open(os.path.join(dir, 'psview_dict.pkl'), 'wb') as f:
            pickle.dump(self.parameterview, f)
        # text for human readability
        with open(os.path.join(dir, 'psview_dict.txt'), 'w') as f:
            json_dump = json.dumps(
                self.parameterview, cls=helpers.NumpyEncoder, indent=2, sort_keys=True)
            f.write(json_dump)

        # sorted list of ranges (if any exist)
        psview_ranges = self.parameterview['custom_params']['ranges']
        ranges = []
        for dic in sorted(psview_ranges.keys()):
            for r in sorted(psview_ranges[dic].keys()):
                ranges.append([dic, r, psview_ranges[dic][r]])
        dim = len(ranges)  # dimension of parameter space
        if dim not in [1, 2]:
            print(
                f'Parameterspace {self.name} has dimension {dim}. ' +
                'Hashes are not printed.')
        else:
            # set up a hash map
            shape = [len(r[2]) for r in ranges]
            hashmap = np.zeros(shape, dtype=object)
            psets = self.parameterview['paramsets']
            for p, h in enumerate(psets.keys()):
                d0_dict, d0_param, d0_range = ranges[0]
                for i, val0 in enumerate(d0_range):
                    if np.all(np.equal(psets[h][d0_dict][d0_param], val0)):
                        if dim == 1:
                            hashmap[i] = h
                        else:
                            d1_dict, d1_param, d1_range = ranges[1]
                            for j, val1 in enumerate(d1_range):
                                if np.all(np.equal(psets[h][d1_dict][d1_param], val1)):
                                    if dim == 2:
                                        hashmap[i][j] = h
                                    else:
                                        d2_dict, d2_param, d2_range = ranges[2]
            if dim == 1:
                hashmap = hashmap.reshape(-1, 1)

            # write ranges and hashmap to file
            ranges_hashmap = {'ranges': ranges, 'hashmap': hashmap}
            dir = os.path.join(self.data_dir, self.name,
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

    def _custom_params_for_parameterview(self, dic, param, value):
        """
        """
        def nested_dict_from_list(keylist, val):
            dic = {keylist[-1]: val}
            for key in reversed(keylist[:-1]):
                dic = {key: dic}
            return dic

        def set_custom_range_or_value(keylist, val, custom_params):
            if isinstance(val, ps.ParameterRange):
                custom_val = list(val)
                custom_type = 'ranges'
            elif isinstance(val, dict):
                for k in val:
                    if isinstance(val[k], ps.ParameterRange):
                        raise Exception(
                            'ParameterRange in nested dict not implemented.')
                custom_val = val
                custom_type = 'values'
            else:
                custom_val = val
                custom_type = 'values'

            custom_dict = {}
            custom_dict[custom_type] = {}
            custom_dict[custom_type][dic] = {}
            custom_dict[custom_type][dic] = nested_dict_from_list(
                keylist, custom_val)

            custom_params = helpers.merge_dictionaries(
                custom_params, custom_dict)
            return

        custom_params = dict(self.parameterview['custom_params'])
        set_custom_range_or_value([param], value, custom_params)
        return custom_params

    def _evaluate_single_parameterset(self, ps_id, paramset):
        """
        Set paths, derive parameters and write jobscripts for this parameter set.

        Parameters
        ----------
        ps_id
            Unique parameter set id.
        paramset
            Parameter set corresponding to ps_id.
        """
        full_data_path = os.path.join(self.data_dir, self.name, ps_id)

        # set paths and create directories for parameters, jobscripts and
        # raw and processed output data
        for dname in \
            ['code', 'parameters', 'jobscripts', 'raw_data', 'processed_data',
             'plots', 'stdout']:
            path = os.path.join(full_data_path, dname)
            if not os.path.isdir(path):
                os.makedirs(path)  # also creates sub directories

        # compute dependent network parameters
        paramset['net_dict'] = helpnet.derive_dependent_parameters(
            paramset['net_dict'])

        # write final parameters to file
        for dic in ['sys_dict', 'sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
            fname = os.path.join(full_data_path, 'parameters', dic)
            # pickle for machine readability
            with open(fname + '.pkl', 'wb') as f:
                pickle.dump(paramset[dic], f)
            # text for human readability
            with open(fname + '.txt', 'w') as f:
                json_dump = json.dumps(
                    paramset[dic], cls=helpers.NumpyEncoder, indent=2, sort_keys=True)
                f.write(json_dump)
        # parameters for NNMT
        nnmt_dic = params_for_neuronal_network_meanfield_tools(
            paramset['net_dict'])
        fname = os.path.join(full_data_path, 'parameters', 'nnmt_dict')
        with open(fname + '.yaml', 'w') as f:
            yaml.dump(nnmt_dic, f, default_flow_style=False)
        # shutil.copyfile(os.path.join('core/parameterization',
        #                             'nnmt_analysis_params.yaml'),
        #                os.path.join(full_data_path, 'parameters',
        #                             'nnmt_ana_dict.yaml'))

        # copy code
        # for d in ['simulation', 'analysis', 'plotting', 'helpers']:
        #    dir_path = os.path.join(full_data_path, 'code', 'core', d)
        #    if not os.path.isdir(dir_path):
        #        os.makedirs(dir_path)

        # filelist = ['run_network.py',
        #            'run_analysis.py',
        #            'run_plotting.py',
        #            'run_lfp_simulation.py',
        #            'run_lfp_postprocess.py',
        #            'run_lfp_plotting.py']

        # for f in filelist:
        #    shutil.copyfile(f, os.path.join(full_data_path, 'code', f))

        # copy 'core' module
        # shutil.copytree('core', os.path.join(full_data_path, 'code', 'core'),
        #                dirs_exist_ok=True)

        # write jobscripts
        self._write_jobscripts(paramset, full_data_path)
        return

    def _write_jobscripts(self, paramset, path):
        """
        Writes a jobscript for each machine (hpc, local) and each step
        (network, analysis, plotting, analyis_and_plotting) specified in the system
        parameters.

        Parameters
        ----------
        paramset
            Parameter set.
        path
            Path to folder of ps_id.
        """
        sys_dict = paramset['sys_dict']
        sim_dict = paramset['sim_dict']

        for machine, dic in sys_dict.items():
            for name, scripts, scriptargs in [['network', ['run_network.py'], ['']],
                                              ['analysis', ['run_analysis.py'], ['']],
                                              ['plotting', ['run_plotting.py'], ['']],
                                              ['analysis_and_plotting', ['run_analysis.py',
                                                                         'run_plotting.py'], [''] * 2],
                                              ['lfp_simulation', ['run_lfp_simulation.py']
                                               * len(self._get_LFP_cell_type_names(path)), self._get_LFP_cell_type_names(path)],
                                              ['lfp_postprocess', [
                    'run_lfp_postprocess.py'], ['']],
                ['lfp_plotting', [
                    'run_lfp_plotting.py'], ['']]
            ]:

                # key of sys_dict defining resources
                res = (name
                       if name in ['network', 'lfp_simulation',
                                   'lfp_postprocess', 'lfp_plotting']
                       else 'analysis_and_plotting')
                dic = sys_dict[machine][res]

                # file for output and errors
                stdout = os.path.join('stdout', name + '.txt')

                # start jobscript
                jobscript = '#!/bin/bash -x\n'

                # define machine specifics
                if machine == 'hpc':
                    # assume SLURM, append resource definitions
                    jobscript += """#SBATCH --job-name=meso
    # SBATCH --partition={}
    # SBATCH --output={}
    # SBATCH --error={}
    # SBATCH --nodes={}
    # SBATCH --ntasks-per-node={}
    # SBATCH --cpus-per-task={}
    # SBATCH --time={}
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    unset DISPLAY
    """
                    run_cmd = 'srun --mpi=pmi2'

                elif machine == 'local':
                    # check which executables are available
                    for mpiexec in ['srun', 'mpiexec', 'mpirun']:
                        out = subprocess.run(
                            ['which', mpiexec], stdout=subprocess.DEVNULL)
                        if out.returncode == 0:
                            if mpiexec in ['mpiexec', 'mpirun']:
                                run_cmd = f'{mpiexec} -n {dic["num_mpi"]}'
                            else:
                                run_cmd = f'{mpiexec} --mpi=pmi2'
                            break
                else:
                    raise NotImplementedError(
                        f'machine {machine} not recognized')

                jobscript += "set -o pipefail\n"

                # append executable(s),
                # number of local threads needed for network simulation,
                # tee output to file for local execution (append for multiple jobs)
                t = dic['local_num_threads'] if name == 'network' else ''
                o_0 = f'2>&1 | tee {stdout}' if machine == 'local' else ''
                o_1 = f'2>&1 | tee -a {stdout}' if machine == 'local' else ''
                if name == 'lfp_plotting':
                    # should be run serially!
                    executables = [
                        f'python3 -u code/{py} {arg} {o_0 if i == 0 else o_1}'
                        for i, (py, arg) in enumerate(zip(scripts, scriptargs))]
                elif name == 'lfp_simulation':
                    executables = []
                    for i, (py, arg) in enumerate(zip(scripts, scriptargs)):
                        y = arg.replace('(', '').replace(')', '')
                        stdout = os.path.join('stdout', f'{name}_{y}.txt')
                        o_0 = f'2>&1 | tee {stdout}' if machine == 'local' else ''
                        o_1 = f'2>&1 | tee -a {stdout}' if machine == 'local' else ''
                        executables += [
                            f'{run_cmd} python3 -u code/{py} "{arg}" {o_0 if i == 0 else o_1}'
                        ]
                elif name == 'lfp_postprocess':
                    executables = [
                        f'{run_cmd} python3 -u code/{py} {arg} {o_0 if i == 0 else o_1}'
                        for i, (py, arg) in enumerate(zip(scripts, scriptargs))]
                else:
                    executables = [
                        f'{run_cmd} python3 -u code/{py} {arg} {t} {o_0 if i == 0 else o_1}'
                        for i, (py, arg) in enumerate(zip(scripts, scriptargs))]
                sep = '\n\n' + 'wait' + '\n\n'
                if name == 'lfp_simulation':
                    # write separate jobscripts for each postsynaptic cell type
                    for i, (executable, arg) in enumerate(zip(executables, scriptargs)):
                        y = arg.replace('(', '').replace(')', '')
                        stdout = os.path.join('stdout', f'{name}_{y}.txt')
                        js = copy.copy(jobscript)
                        js += executable
                        if machine == 'hpc':
                            if type(dic['wall_clock_time']) in [str]:
                                wt = dic['wall_clock_time']
                            elif type(dic['wall_clock_time']) in [list, tuple]:
                                # compute walltime specific to each cell type y:
                                _wt = int(round((sim_dict['t_presim'] + sim_dict['t_sim']) / 1E3 *
                                                dic['wall_clock_time'][i] * 1.5
                                                ))  # add 50% buffer
                                wt = strftime("%H:%M:%S", gmtime(_wt))
                            else:
                                raise Exception(
                                    f"wall_clock_time={dic['wall_clock_time']} must be str or list/tuple")
                            # fill in work string
                            js = js.format(
                                dic['partition'],
                                stdout,
                                stdout,
                                dic['num_nodes'],
                                dic['num_mpi_per_node'],
                                dic['local_num_threads'],
                                wt  # dic['wall_clock_time']
                            )
                        y = arg.replace('(', '').replace(')', '')
                        fname = os.path.join(
                            path, 'jobscripts', f"{machine}_{name}_{y}.sh")
                        with open(fname, 'w') as f:
                            f.write(js)
                else:
                    jobscript += sep.join(executables)
                    if machine == 'hpc':
                        jobscript = jobscript.format(
                            dic['partition'],
                            stdout,
                            stdout,
                            dic['num_nodes'],
                            dic['num_mpi_per_node'],
                            dic['local_num_threads'],
                            dic['wall_clock_time']
                        )

                    # write jobscript
                    fname = os.path.join(path, 'jobscripts',
                                         f"{machine}_{name}.sh")
                    with open(fname, 'w') as f:
                        f.write(jobscript)
        return

    def _get_LFP_cell_type_names(self, path):
        """
        Returns a list of LFP cell type names.

        Parameters
        ----------
        path
            Absolute path of parameter set.
        """
        path_lfp_data = 'lfp'
        dics = []
        for dic in ['sim_dict', 'net_dict']:
            with open(f'{path}/parameters/{dic}.pkl', 'rb') as f:
                dics.append(pickle.load(f))
        PS = lfp_parameters.get_parameters(path_lfp_data=path_lfp_data,
                                           sim_dict=dics[0],
                                           net_dict=dics[1])
        return PS.y


def params_for_neuronal_network_meanfield_tools(net_dict):
    """
    Creates a dictionary with parameters for mean-field theoretical analysis
    with NNMT - the Neuronal Network Meanfield Toolbox
    (https://github.com/INM-6/nnmt).

    The parameters for the full network are used.
    A normally distributed delay is assumed.
    Since NNMT only allows for one synaptic time constant, the default one is
    used.

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

    # reverse weight scaling with synaptic time constant
    w = (net_dict['full_weight_matrix_mean'][0][0] *
         net_dict['neuron_params']['tau_syn_ex'] /
         net_dict['neuron_params']['tau_syn_default']).astype(list)

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
        'tau_s': {'val': net_dict['neuron_params']['tau_syn_default'],
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
        'w': {'val': w,
              'unit': 'pA'},
        'w_ext': {'val': w,
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


def run_parametersets(
        func,
        parameterview=None,
        paramspace_keys=None,
        with_base_params=False,
        ps_ids=[],
        data_dir=None,
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


def run_single_jobs(paramspace_key, ps_id, data_dir=None,
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

    # clean exit in case of no jobs
    if len(jobs) == 0:
        return

    jobinfo = ' and '.join(jobs) if len(jobs) > 1 else jobs[0]
    info = f'{jobinfo} for {paramspace_key} - {ps_id}.'

    def submit_lfp_simulation_jobs(dependency=None):
        # these jobs can run in parallel
        lfp_job_scripts = []
        for y in self._get_LFP_cell_type_names(full_data_path):
            y = y.replace('(', '').replace(')', '')
            lfp_job_scripts.append(f'hpc_lfp_simulation_{y}.sh')
        jobid = []  # job == lfp_postprocess require all lfp_simulation jobs to have finished
        for js in lfp_job_scripts:
            if dependency is None:
                submit = f'sbatch --account $BUDGET_ACCOUNTS jobscripts/{js}'
            else:
                submit = (
                    f'sbatch --account $BUDGET_ACCOUNTS ' +
                    f'--dependency=afterok:{dependency} jobscripts/{js}'
                )
            output = subprocess.getoutput(submit)
            print(output, submit)
            jobid.append(output.split(' ')[-1])
        return jobid

    if machine == 'hpc':
        print('Submitting ' + info)
        if jobs[0] == 'lfp_simulation':
            jobid = submit_lfp_simulation_jobs(dependency=None)
        else:
            submit = f'sbatch --account $BUDGET_ACCOUNTS jobscripts/{machine}_{jobs[0]}.sh'
            output = subprocess.getoutput(submit)
            print(output, submit)
            jobid = output.split(' ')[-1]
        # submit any subsequent jobs with dependency
        if len(jobs) > 1:
            for i, job in enumerate(jobs[1:]):
                if job == 'lfp_simulation':
                    jobid = submit_lfp_simulation_jobs(dependency=jobid)
                elif job == 'lfp_postprocess':
                    # has multiple dependencies
                    if isinstance(jobid, (list, tuple)):
                        afterok = ':'.join(jobid)
                    else:
                        afterok = jobid
                    submit = (
                        f'sbatch --account $BUDGET_ACCOUNTS ' +
                        f'--dependency=afterok:{afterok} jobscripts/{machine}_{job}.sh'
                    )
                    output = subprocess.getoutput(submit)
                    print(output, submit)
                    jobid = output.split(' ')[-1]
                else:
                    submit = (
                        f'sbatch --account $BUDGET_ACCOUNTS ' +
                        f'--dependency=afterok:{jobid} jobscripts/{machine}_{job}.sh'
                    )
                    output = subprocess.getoutput(submit)
                    print(output, submit)
                    jobid = output.split(' ')[-1]

    elif machine == 'local':
        print('Running ' + info)
        for job in jobs:
            if job == 'lfp_simulation':
                for y in self.get_LFP_cell_type_names(full_data_path):
                    y = y.replace('(', '').replace(')', '')
                    retval = os.system(
                        f'bash jobscripts/{machine}_{job}_{y}.sh')
                    if retval != 0:
                        raise Exception(f"os.system failed: {retval}")
            else:
                retval = os.system(f'bash jobscripts/{machine}_{job}.sh')
                if retval != 0:
                    raise Exception(f"os.system failed: {retval}")

    os.chdir(cwd)
    return


def run_single_nnmt(paramspace_key, ps_id, data_dir=None):
    """
    Computes some theoretical quantities with NNMT for a single
    parameter set.

    NNMT is the Neuronal Network Meanfield Toolbox
    (https://github.com/INM-6/nnmt).

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
    import nnmt

    print(f'Computing theory for {paramspace_key} - {ps_id}.')

    os.chdir(os.path.join(data_dir, paramspace_key, ps_id))

    # nnmt network object of type Microcircuit
    nw = nnmt.models.Microcircuit(
        network_params=os.path.join(
            'parameters', 'nnmt_dict.yaml'), analysis_params=os.path.join(
            'parameters', 'nnmt_ana_dict.yaml'))

    # working point for exponentially shape post synaptic currents
    wp = nnmt.lif.exp.working_point(nw)
    # transfer function
    nnmt.lif.exp.transfer_function(nw)
    # delay distribution matrix
    nnmt.network_properties.delay_dist_matrix(nw)
    # effective connectivity matrix
    nnmt.lif.exp.effective_connectivity(nw)
    # power spectra
    power = nnmt.lif.exp.power_spectra(nw)
    freqs = nw.analysis_params['omegas'] / (2. * np.pi)

    # sensitivity measure
    sensitivity_dict = nnmt.lif.exp.sensitivity_measure_all_eigenmodes(nw)

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
        sensitivity=sensitivity_dict)

    return
