"""Mesocircuit framework
------------------------

Parameterspace evaluation and handling of job execution.

Definition of the main classes MesocircuitExperiment and Mesocircuit.
"""

import mesocircuit
from mesocircuit.parameterization import helpers_network as helpnet
from mesocircuit.parameterization import helpers_analysis as helpana
import os
import sys
import subprocess
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
    name_exp
        Name of the experiment. All corresponding data and scripts will be
        written to a folder with this name.
    custom_params : dict, optional
        Dictionary with new parameters or parameter ranges to overwrite the
        default ones.
    data_dir
        Absolute path to write data to.
    load
        If True, parameters are not newly evaluated and the earlier saved
        parameterview and Mesocircuit(s) are loaded.
    """

    def __init__(self, name_exp='base', custom_params=None, data_dir=None,
                 load=False):
        """
        Instantiating.
        """
        self.name_exp = name_exp
        print(f'Instantiating MesocircuitExperiment: {self.name_exp}')

        # data directory
        if not data_dir:
            self.data_dir = self._auto_data_directory()
        else:
            self.data_dir = data_dir
        self.data_dir_exp = os.path.join(self.data_dir, self.name_exp)

        # check if data directory exists
        if not os.path.isdir(self.data_dir_exp):
            print(f'creating directory {self.data_dir_exp}')
            os.makedirs(self.data_dir_exp)
        
        print(f'Data directory: {self.data_dir_exp}')

        if not load:
            self.parameterview, self.circuits = \
                self._evaluate_parameters(custom_params)
        else:
            fn = os.path.join(self.data_dir_exp, 'parameter_space',
                              'parameters', 'psview_dict.pkl')
            print(fn)
            with open(fn, 'rb') as f:
                ps_view = pickle.load(f)
                self.parameterview = ps_view
            ps_ids = self.parameterview['paramsets'].keys()
            self.circuits = [
                Mesocircuit(self.data_dir,
                            self.name_exp,
                            ps_id,
                            load_parameters=True) for ps_id in ps_ids]

    def _auto_data_directory(self, dirname='mesocircuit_data'):
        """
        Automatically determines a data directory.

        Parameters
        ----------
        dirname
            Name of data directory.
        """
        try:
            data_dir = os.path.join(os.environ['SCRATCH'],
                                    os.environ['USER'],
                                    dirname)
        except BaseException:
            data_dir = os.path.join(os.getcwd(), dirname)
        return data_dir

    def _evaluate_parameters(self, custom_params={}):
        """
        Evaluates parameters and creates a parameter view for the experiment.

        Parameters
        ----------
        custom_params : dict, optional
            Dictionary with new parameters or parameter ranges to overwrite the
            default ones.

        Returns
        -------
        parameterview
            Overview of parameter spaces and corresponding IDs.
        circuits
            Mesocircuit objects.
        """
        # parameterspaces built with the parameters module
        parameterspaces = ps.ParameterSpace({})
        # overview of parameter spaces and corresponding
        parameterview = {}
        parameterview['custom_params'] = {}
        parameterview['custom_params']['ranges'] = {}
        parameterview['custom_params']['values'] = {}

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
                    parameterview['custom_params'] = \
                        self._custom_params_for_parameterview(
                            parameterview['custom_params'], dic, param, value)

        # only sim_dict and net_dict are used to compute a unique id
        dicts_unique = ['sim_dict', 'net_dict']
        sub_paramspace = ps.ParameterSpace(
            {k: parameterspaces[k] for k in dicts_unique})

        parameterview['paramsets'] = {}
        circuits = []
        for sub_paramset in sub_paramspace.iter_inner():
            ps_id = helpers.get_unique_id(sub_paramset)

            # add ana_dict and plot_dict to get full paramset
            # (deep copy of sub_paramset is needed)
            paramset = {
                **copy.deepcopy(sub_paramset),
                'sys_dict': parameterspaces['sys_dict'],
                'ana_dict': parameterspaces['ana_dict'],
                'plot_dict': parameterspaces['plot_dict']}

            # add parameterset values of ranges to parameterview
            parameterview['paramsets'][ps_id] = {}
            for dic in parameterview['custom_params']['ranges']:
                parameterview['paramsets'][ps_id][dic] = {}
                for param, val in \
                        parameterview['custom_params']['ranges'][dic].items():
                    parameterview[
                        'paramsets'][ps_id][dic][param] = paramset[dic][param]

            # instantiate a Mesocircuit object
            circuit = Mesocircuit(self.data_dir, self.name_exp, ps_id)

            # evaluate the parameter set
            circuit._evaluate_parameterset(paramset)

            circuits.append(circuit)

        # setup for parameterspace analysis
        for dname in ['parameters', 'plots']:
            path = os.path.join(
                self.data_dir_exp, 'parameter_space', dname)
            if not os.path.isdir(path):
                os.makedirs(path)

        # write parameterview to file
        dir = os.path.join(self.data_dir_exp, 'parameter_space', 'parameters')
        # pickle for machine readability
        with open(os.path.join(dir, 'psview_dict.pkl'), 'wb') as f:
            pickle.dump(parameterview, f)
        # text for human readability
        with open(os.path.join(dir, 'psview_dict.txt'), 'w') as f:
            json_dump = json.dumps(
                parameterview, cls=helpers.NumpyEncoder, indent=2, sort_keys=True)
            f.write(json_dump)

        # sorted list of ranges (if any exist)
        psview_ranges = parameterview['custom_params']['ranges']
        ranges = []
        for dic in sorted(psview_ranges.keys()):
            for r in sorted(psview_ranges[dic].keys()):
                ranges.append([dic, r, psview_ranges[dic][r]])
        dim = len(ranges)  # dimension of parameter space
        if dim not in [1, 2]:
            print(
                f'Parameterspace {self.name_exp} has dimension {dim}. ' +
                'Hashes are not printed.')
        else:
            # set up a hash map
            shape = [len(r[2]) for r in ranges]
            hashmap = np.zeros(shape, dtype=object)
            psets = parameterview['paramsets']
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
            if dim == 1:
                hashmap = hashmap.reshape(-1, 1)

            # write ranges and hashmap to file
            ranges_hashmap = {'ranges': ranges, 'hashmap': hashmap}
            dir = os.path.join(
                self.data_dir_exp, 'parameter_space', 'parameters')
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
        return parameterview, circuits

    def _custom_params_for_parameterview(self, old_custom_params, dic, param, value):
        """
        Handles custom parameters for parameter view by merging dictionaries.

        Parameters
        ----------
        old_custom_params
            Dictionary with all old custom parameters.
        dic
            Parameter dictionary name.
        param
            New parameter to be added.
        value
            Value of new parameter.

        Returns
        -------
        custom_params
            Updated dictionary with custom parameters.
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

        custom_params = dict(old_custom_params)
        set_custom_range_or_value([param], value, custom_params)
        return custom_params


class Mesocircuit():
    """
    Mesocircuit class for handling a single parameter set.

    Parameters
    ----------
    data_dir_exp
        Absolute path to directory of corresponding MesocircuitExperiment.
    name_exp
        Name of the MesocircuitExperiment.
    ps_id
        Unique parameter set id.
    load_parameters
        If True, load parameters from file. Sets class attributes for each
        dictionary.
    """

    def __init__(self, data_dir='', name_exp='', ps_id='', load_parameters=False):
        """
        Instantiating.
        """
        self.data_dir = data_dir
        self.name_exp = name_exp
        self.ps_id = ps_id
        self.data_dir_circuit = os.path.join(data_dir, name_exp, ps_id)

        if load_parameters:
            path = os.path.join(self.data_dir_circuit, 'parameters')
            dics = []
            for dic in ['sys_dict', 'sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
                with open(os.path.join(path, f'{dic}.pkl'), 'rb') as f:
                    dics.append(pickle.load(f))
            self.sys_dict, self.sim_dict, self.net_dict, self.ana_dict, self.plot_dict = dics

    def _evaluate_parameterset(self, paramset):
        """
        Derive parameters and write jobscripts.

        Parameters
        ----------
        paramset
            Parameter set corresponding to ps_id.

        Returns
        -------
            Updated parameter set.
        """
        print(f'Evaluating parameters for ps_id: {self.ps_id}')
        # set paths and create directories for parameters, jobscripts and
        # raw and processed output data
        for dname in \
            ['parameters', 'jobscripts', 'raw_data', 'processed_data',
             'plots', 'stdout']:
            path = os.path.join(self.data_dir_circuit, dname)
            if not os.path.isdir(path):
                os.makedirs(path)  # also creates sub directories

        # compute dependent network parameters
        paramset['net_dict'] = helpnet.derive_dependent_parameters(
            paramset['net_dict'])

        # compute dependent analysis parameters
        paramset['ana_dict'] = helpana.derive_dependent_parameters(
            paramset['net_dict'], paramset['sim_dict'], paramset['ana_dict'])

        # write final parameters to file
        for dic in ['sys_dict', 'sim_dict', 'net_dict', 'ana_dict', 'plot_dict']:
            fname = os.path.join(self.data_dir_circuit, 'parameters', dic)
            # pickle for machine readability
            with open(fname + '.pkl', 'wb') as f:
                pickle.dump(paramset[dic], f)
            # text for human readability
            with open(fname + '.txt', 'w') as f:
                json_dump = json.dumps(
                    paramset[dic], cls=helpers.NumpyEncoder, indent=2, sort_keys=True)
                f.write(json_dump)
        # parameters for NNMT
        nnmt_dic = self._params_for_neuronal_network_meanfield_tools(
            paramset['net_dict'])
        fname = os.path.join(self.data_dir_circuit, 'parameters', 'nnmt_dict')
        with open(fname + '.yaml', 'w') as f:
            yaml.dump(nnmt_dic, f, default_flow_style=False)

        # store parameters as class attributes
        self.sys_dict = paramset['sys_dict']
        self.sim_dict = paramset['sim_dict']
        self.net_dict = paramset['net_dict']
        self.ana_dict = paramset['ana_dict']
        self.plot_dict = paramset['plot_dict']

        # write jobscripts
        self._write_jobscripts(paramset, self.data_dir_circuit)
        return paramset

    def _params_for_neuronal_network_meanfield_tools(self, net_dict):
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
            # no thalamus
            'populations': net_dict['populations'][:-1].tolist(),
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

        # path to run_scripts
        run_path = os.path.join(os.path.dirname(mesocircuit.__file__), 'run')

        sys_dict = paramset['sys_dict']
        sim_dict = paramset['sim_dict']

        # generic arguments for all run_scripts pointing to right circuit,
        # should be last argument
        a = '$DATA_DIR $NAME_EXP $PS_ID'

        for machine, dic in sys_dict.items():

            # local_num_threads for network simulation
            t = str(sys_dict[machine]['network']['local_num_threads'])

            LFP_cells = self._get_LFP_cell_type_names(path)
            lfp_arg = [c + ' ' + a for c in LFP_cells]

            for name, scripts, scriptargs in [
                ['network', ['run_network.py'], [t + ' ' + a]],
                ['analysis', ['run_analysis.py'], [a]],
                ['plotting', ['run_plotting.py'], [a]],
                ['analysis_and_plotting', ['run_analysis.py',
                                           'run_plotting.py'], [a] * 2],
                ['lfp_simulation', ['run_lfp_simulation.py']
                 * len(LFP_cells), lfp_arg],
                ['lfp_postprocess', ['run_lfp_postprocess.py'], [a]],
                    ['lfp_plotting', ['run_lfp_plotting.py'], [a]]]:

                # LFP simulation not implemented for microcircuit
                if (self.name_exp.rfind('microcircuit') >= 0) & (name.rfind('lfp') >= 0):
                    continue

                # key of sys_dict defining resources
                res = (name
                       if name in ['network', 'lfp_simulation',
                                   'lfp_postprocess', 'lfp_plotting']
                       else 'analysis_and_plotting')
                dic = sys_dict[machine][res]

                # start jobscript
                jobscript = '#!/bin/bash -x\n'

                # define machine specifics
                if machine == 'hpc':
                    # assume SLURM, append resource definitions
                    # the following could be added:
                    # export OMP_DISPLAY_ENV=VERBOSE
                    # export OMP_DISPLAY_AFFINITY=TRUE
                    jobscript += """#SBATCH --job-name=meso
#SBATCH --partition={}
#SBATCH --output={}
#SBATCH --error={}
#SBATCH --nodes={}
#SBATCH --ntasks-per-node={}
#SBATCH --time={}
export NUMEXPR_MAX_THREADS={}
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS={}
unset DISPLAY
"""

                    if name == 'network':
                        # the allocator jemalloc is here used for improved
                        # performance (see Ippen et al., 2017):
                        # get path to jemalloc
                        # "which jemalloc" executed on the command line returns
                        # something like
                        # '/p/software/jurecadc/stages/2022/software/jemalloc/5.2.1-GCCcore-11.2.0/bin/jemalloc.sh'
                        try:
                            which_jemalloc = subprocess.check_output(
                                ["which", "jemalloc.sh"]).decode(sys.stdout.encoding).strip()
                            # replace '/bin/jemalloc.sh' by '/lib64/libjemalloc.so'
                            jemalloc_path = ('/').join(
                                which_jemalloc.split('/')[:-2]) + \
                                '/lib64/libjemalloc.so'
                            jobscript += f'export LD_PRELOAD={jemalloc_path}\n'
                        except:
                            print(
                                "LD_PRELOAD skipped because jemalloc is not in PATH.")

                    if name in ['lfp_simulation', 'lfp_postprocess', 'lfp_plotting']:
                        run_cmd = f'srun --mpi=pmi2'
                    else:
                        run_cmd = f'srun --cpus-per-task={dic["local_num_threads"]} --threads-per-core=1 --cpu-bind=rank'

                elif machine == 'local':
                    # check which executables are available
                    for mpiexec in ['srun', 'mpiexec', 'mpirun']:
                        out = subprocess.run(
                            ['which', mpiexec], stdout=subprocess.DEVNULL)
                        if out.returncode == 0:
                            if mpiexec in ['mpiexec', 'mpirun']:
                                run_cmd = f'{mpiexec} -n {dic["num_mpi"]}'
                            else:
                                run_cmd = f'{mpiexec}'
                            break
                    # get rid of annoying openmpi segfault on macOS
                    if sys.platform == 'darwin':
                        run_cmd = 'export TMPDIR=/tmp\n' + run_cmd
                else:
                    raise NotImplementedError(
                        f'machine {machine} not recognized')

                jobscript += "set -o pipefail\n"
                jobscript += f"RUN_PATH={run_path}\n"
                jobscript += f"DATA_DIR={self.data_dir}\n"
                jobscript += f"NAME_EXP={self.name_exp}\n"
                jobscript += f"PS_ID={self.ps_id}\n"

                # file for output and errors (for batch scripts of hpc the
                # evalutated path is needed)
                stdout = f"$DATA_DIR/$NAME_EXP/$PS_ID/stdout/{name}.txt"
                stdout_hpc = os.path.join(
                    self.data_dir_circuit, 'stdout', name + '.txt')

                # append executable(s),
                # tee output to file for local execution (append for multiple jobs)
                o_0 = f'2>&1 | tee {stdout}' if machine == 'local' else ''
                o_1 = f'2>&1 | tee -a {stdout}' if machine == 'local' else ''
                if name == 'lfp_plotting':
                    # should be run serially!
                    executables = [
                        f'python3 -u $RUN_PATH/{py} {arg} {o_0 if i == 0 else o_1}'
                        for i, (py, arg) in enumerate(zip(scripts, scriptargs))]
                elif name == 'lfp_simulation':
                    executables = []
                    for i, (py, arg) in enumerate(zip(scripts, scriptargs)):
                        # split cell types (y) from other argumets
                        # cell types shall be printed in ""
                        y = arg.split(' ')[0]
                        y_ = y.replace('(', '').replace(')', '')
                        variables = arg.split(' ')[1:]
                        variables = ' '.join(variables)
                        stdout = os.path.join(
                            self.data_dir_circuit, 'stdout', f'{name}_{y_}.txt')
                        o_0 = f'2>&1 | tee {stdout}' if machine == 'local' else ''
                        o_1 = f'2>&1 | tee -a {stdout}' if machine == 'local' else ''
                        executables += [
                            f'{run_cmd} python3 -u $RUN_PATH/{py} "{y}" {variables} {o_0 if i == 0 else o_1}'
                        ]
                elif name == 'lfp_postprocess':
                    executables = [
                        f'{run_cmd} python3 -u $RUN_PATH/{py} {arg} {o_0 if i == 0 else o_1}'
                        for i, (py, arg) in enumerate(zip(scripts, scriptargs))]
                else:
                    executables = [
                        f'{run_cmd} python3 -u $RUN_PATH/{py} {arg} {o_0 if i == 0 else o_1}'
                        for i, (py, arg) in enumerate(zip(scripts, scriptargs))]
                sep = '\n\n' + 'wait' + '\n\n'
                if name == 'lfp_simulation':
                    # write separate jobscripts for each postsynaptic cell type
                    for i, (executable, arg) in enumerate(zip(executables, scriptargs)):
                        y = arg.split(' ')[0]
                        y = y.replace('(', '').replace(')', '')
                        stdout = os.path.join(
                            self.data_dir_circuit, 'stdout', f'{name}_{y}.txt')
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
                                wt,  # dic['wall_clock_time']
                                dic['max_num_cores'],
                                dic['local_num_threads']
                            )
                        y = arg.split(' ')[0]
                        y = y.replace('(', '').replace(')', '')
                        fname = os.path.join(
                            path, 'jobscripts', f"{machine}_{name}_{y}.sh")
                        with open(fname, 'w') as f:
                            f.write(js)

                else:
                    jobscript += sep.join(executables)
                    if machine == 'hpc':
                        jobscript = jobscript.format(
                            dic['partition'],
                            stdout_hpc,
                            stdout_hpc,
                            dic['num_nodes'],
                            dic['num_mpi_per_node'],
                            dic['wall_clock_time'],
                            dic['max_num_cores'],
                            dic['local_num_threads']
                        )

                    # write jobscript
                    fname = os.path.join(path, 'jobscripts',
                                         f"{machine}_{name}.sh")
                    with open(fname, 'w') as f:
                        f.write(jobscript)
        return

    def run_jobs(self, jobs=['network', 'analysis_and_plotting'], machine='hpc'):
        """
        Runs jobs of a single parameter set.

        Parameters
        ----------
        jobs
            List of one or multiple of 'network, 'analysis, 'plotting', and
            'anlysis_and_plotting'.
        machine
            'local' or 'hpc'.
        """
        # clean exit in case of no jobs
        if len(jobs) == 0:
            return

        # prune LFP jobs if running the microcircuit model
        if self.name_exp.rfind('microcircuit') >= 0:
            jobs = [x for x in jobs if x.rfind('lfp') < 0]

        jobinfo = ' and '.join(jobs) if len(jobs) > 1 else jobs[0]
        info = f'{jobinfo} for {self.name_exp} - {self.ps_id}.'

        dir_jobscripts = os.path.join(self.data_dir_circuit, 'jobscripts')

        def submit_lfp_simulation_jobs(dependency=None):
            # these jobs can run in parallel
            lfp_job_scripts = []
            for y in self._get_LFP_cell_type_names(self.data_dir_circuit):
                y = y.replace('(', '').replace(')', '')
                lfp_job_scripts.append(f'hpc_lfp_simulation_{y}.sh')
            jobid = []  # job == lfp_postprocess require all lfp_simulation jobs to have finished
            for js in lfp_job_scripts:
                if dependency is None:
                    submit = f'sbatch --account $BUDGET_ACCOUNTS {dir_jobscripts}/{js}'
                else:
                    submit = (
                        f'sbatch --account $BUDGET_ACCOUNTS ' +
                        f'--dependency=afterok:{dependency} {dir_jobscripts}/{js}'
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
                submit = f'sbatch --account $BUDGET_ACCOUNTS {dir_jobscripts}/{machine}_{jobs[0]}.sh'
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
                            f'--dependency=afterok:{afterok} {dir_jobscripts}/{machine}_{job}.sh'
                        )
                        output = subprocess.getoutput(submit)
                        print(output, submit)
                        jobid = output.split(' ')[-1]
                    else:
                        submit = (
                            f'sbatch --account $BUDGET_ACCOUNTS ' +
                            f'--dependency=afterok:{jobid} {dir_jobscripts}/{machine}_{job}.sh'
                        )
                        output = subprocess.getoutput(submit)
                        print(output, submit)
                        jobid = output.split(' ')[-1]

        elif machine == 'local':
            print('Running ' + info)
            for job in jobs:
                if job == 'lfp_simulation':
                    for y in self._get_LFP_cell_type_names(self.data_dir_circuit):
                        y = y.replace('(', '').replace(')', '')
                        retval = os.system(
                            f'bash {dir_jobscripts}/{machine}_{job}_{y}.sh')
                        if retval != 0:
                            raise Exception(f"os.system failed: {retval}")
                else:
                    retval = os.system(
                        f'bash {dir_jobscripts}/{machine}_{job}.sh')
                    if retval != 0:
                        raise Exception(f"os.system failed: {retval}")

        return
