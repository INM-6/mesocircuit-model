"""PyNEST Mesocircuit: Network Class
------------------------------------

Main file of the mesocircuit defining the ``Network`` class with functions to
build and simulate the network.

"""

import os
import numpy as np
from pathlib import Path
import tarfile
import h5py
import nest
from mpi4py import MPI
from ..helpers.mpiops import GathervRecordArray
import multiprocessing as mp


class Network:
    """ Provides functions to setup NEST, to create and connect all nodes of
    the network and to simulate.

    Instantiating a Network object derives dependent parameters and already
    initializes the NEST kernel.

    Parameters
    ---------
    sim_dict
        Dictionary containing all parameters specific to the simulation
        (derived from: ``base_sim_params.py``).
    net_dict
         Dictionary containing all parameters specific to the neuron and
         network models (derived from: ``base_network_params.py``).
    """

    def __init__(self, sim_dict, net_dict, local_num_threads):
        self.sim_dict = sim_dict
        self.net_dict = net_dict

        # wipe files from raw output directory it they exist
        self.__wipe()

        # check parameters and print information
        self.__check_parameters()

        # initialize the NEST kernel
        self.__setup_nest(local_num_threads)
        return

    def create(self):
        """ Creates all network nodes.

        Neuronal populations and recording and stimulating devices are created.

        """
        self.__create_neuronal_populations()
        if len(self.sim_dict['rec_dev']) > 0:
            self.__create_recording_devices()
        if self.net_dict['poisson_input']:
            self.__create_poisson_bg_input()
        if self.net_dict['thalamic_input']:
            self.__create_thalamic_stim_input()
        if self.net_dict['dc_input']:
            self.__create_dc_stim_input()
        return

    def connect(self):
        """ Connects the network.

        Recurrent connections among neurons of the neuronal populations are
        established, and recording and stimulating devices are connected.

        The ``self.__connect_*()`` functions use ``nest.Connect()`` calls which
        set up the postsynaptic connectivity.
        Since the introduction of the 5g kernel in NEST 2.16.0 the full
        connection infrastructure including presynaptic connectivity is set up
        afterwards in the preparation phase of the simulation.
        The preparation phase is usually induced by the first
        ``nest.Simulate()`` call.
        For including this phase in measurements of the connection time,
        we induce it here explicitly by calling ``nest.Prepare()``.

        """
        self.__connect_neuronal_populations()

        if len(self.sim_dict['rec_dev']) > 0:
            self.__connect_recording_devices()
        if self.net_dict['poisson_input']:
            self.__connect_poisson_bg_input()
        if self.net_dict['thalamic_input']:
            self.__connect_thalamic_stim_input()
        if self.net_dict['dc_input']:
            self.__connect_dc_stim_input()

        nest.Prepare()
        nest.Cleanup()
        return

    def presimulate(self, t_presim):
        """
        Simulates the mesocircuit for a pre-simulation time.

        data_prefix is set such that the following simulation does not
        overwrite data recorded during the presimulation time.

        Parameters
        ----------
        t_presim
            Pre-simulation time (in ms).
        """
        if nest.Rank() == 0:
            print('Pre-simulating {} ms.'.format(t_presim))

        nest.SetKernelStatus({'data_prefix': 'presim_'})
        nest.Simulate(t_presim)

        return

    def simulate(self, t_sim):
        """
        Simulates the mesocircuit for a simulation time.

        Parameters
        ----------
        t_sim
            Simulation time (in ms).
        """
        if nest.Rank() == 0:
            print('Simulating {} ms.'.format(t_sim))

        nest.SetKernelStatus({'data_prefix': 'sim_'})
        nest.Simulate(t_sim)

        # dump recorded spikes to HDF5 file
        self.__write_spikes(fname='spike_recorder.h5')

        return

    def tar_raw_data(self,
                     delete_files=True,
                     filepatterns=['*.dat'],
                     mode='w',
                     method='safe'):
        '''
        Create tar file of content in `raw_data/<>` and optionally
        delete files matching given pattern.

        Parameters
        ----------
        output_path: path
            params.raw_nest_output_path
        delete_files: bool
            if True, delete files matching pattern
        filepatterns: list of str
            patterns of files being deleted
        mode: String
            tarfile.open file mode. Default: 'w'
        method: str
            if method=='perl' use faster perl method,
            if method=='safe' (default) use safer Path.unlink method.
        '''
        output_path = 'raw_data'

        if nest.Rank() == 0:
            # create tarfile
            fname = output_path + '.tar'
            with tarfile.open(fname, mode) as t:
                t.add(output_path,
                      arcname=os.path.split(output_path)[-1])
        MPI.COMM_WORLD.Barrier()

        # remove files from <raw_nest_output_path>
        if delete_files:
            for pattern in filepatterns:
                self.__wipe(pattern, method=method)

        return

    def __wipe(self, pattern='*', method='safe'):
        """ Wipe raw output directory from any existing files.
        Will create folder if it does not exist.

        Parameters
        ----------
        pattern: str
            file pattern to remove. Default '*'
        method: str
            if method=='perl' use faster perl method,
            if method=='safe' (default) use safer Path.unlink method.
        """
        output_path = self.sim_dict["path_raw_data"]
        if nest.Rank() == 0:
            if os.path.isdir(output_path):
                if method == 'perl':
                    cwd = os.getcwd()
                    os.chdir(output_path)
                    os.system('perl -e "for(<' + '{}'.format(pattern) +
                              '>){((stat)[9]<(unlink))}"')
                    os.chdir(cwd)
                elif method == 'safe':
                    for p in Path(output_path).glob(pattern):
                        while p.is_file():
                            try:
                                p.unlink()
                            except OSError as e:
                                print('Error: {} : {}'.format(p, e.strerror))
                # remove raw directory
                os.rmdir(output_path)

        MPI.COMM_WORLD.Barrier()
        return

    def __write_spikes(self, fname='spike_recorder.h5'):
        """
        Writes recorded spikes from memory to HDF5 file.

        Parameters
        ----------
        fname: str
            Output file name. Path to raw data folder will be prepended

        """
        fn = os.path.join('raw_data', fname)
        if nest.Rank() == 0:
            f = h5py.File(fn, 'w')
        for i, (label, sr) in enumerate(zip(self.net_dict['populations'],
                                            self.spike_recorders)):
            events = nest.GetStatus(sr)[0]['events']
            names = ['nodeid', 'time_ms']
            formats = ['i4', 'f8']
            data = np.recarray((events['senders'].size),
                               names=names, formats=formats)
            data['nodeid'] = events['senders']
            data['time_ms'] = events['times']

            DATA = GathervRecordArray(data)

            if nest.Rank() == 0:
                f[label] = DATA

        if nest.Rank() == 0:
            f.close()

    def __wipe(self):
        """ Wipes raw output directory from any existing files"""
        if nest.Rank() == 0:
            if os.path.isdir('raw_data'):
                for p in Path('raw_data').glob('*'):
                    while p.is_file():
                        try:
                            p.unlink()
                        except OSError as e:
                            print('Error: {} : {}'.format(p, e.strerror))
        MPI.COMM_WORLD.Barrier()
        return

    def __check_parameters(self):
        """
        Checks parameters and prints information.
        In the current implementation only a message specifying the neuron
        and indegree scaling is printed.
        """

        if nest.Rank() == 0:
            message = ''
            if self.net_dict['N_scaling'] != 1:
                message += \
                    'Neuron numbers are scaled by a factor of {:.3f}.\n'.format(
                        self.net_dict['N_scaling'])
            if self.net_dict['K_scaling'] != 1:
                message += \
                    'Indegrees are scaled by a factor of {:.3f}.'.format(
                        self.net_dict['K_scaling'])
                message += '\n  Weights and DC input are adjusted to compensate.\n'
            print(message)
        return

    def __setup_nest(self, local_num_threads):
        """ Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        The number of seeds for random number generation are computed based on
        the total number of virtual processes
        (number of MPI processes x number of threads per MPI process).

        Parameters
        ----------
        local_num_threads
            Number of threads per MPI process. If 'auto', an adequate number is
            inferred.
        """
        nest.ResetKernel()

        # automatically set thread number such that the the total number of
        # virtual processes does not exceed the number of available logical
        # cores
        if local_num_threads == 'auto':
            nproc = mp.cpu_count() // 2  # disable multithreading
            local_threads = int(nproc / nest.GetKernelStatus('num_processes'))
        else:
            local_threads = int(local_num_threads)

        nest.SetKernelStatus(
            {'local_num_threads': local_threads})
        N_vp = nest.GetKernelStatus('total_num_virtual_procs')

        # set seeds for random number generation
        master_seed = self.sim_dict['master_seed']
        grng_seed = master_seed + N_vp
        rng_seeds = (master_seed + N_vp + 1 + np.arange(N_vp)).tolist()

        if nest.Rank() == 0:
            print('Master seed: {} '.format(master_seed))
            print('  Total number of virtual processes: {}'.format(N_vp))
            print('  Global random number generator seed: {}'.format(grng_seed))
            print(
                '  Seeds for random number generators of virtual processes (old): ' +
                '{}'.format(rng_seeds))

        # pass parameters to NEST kernel

        # TODO adjust code on random number generator seeds if PR #1549 is merged
        if 'grng_seed' in nest.GetKernelStatus(): # in NEST 2
            rng_seed = 'grng_seed'
        else:
            rng_seed = 'rng_seed'

        self.sim_resolution = self.sim_dict['sim_resolution']
        kernel_dict = {
            'resolution': self.sim_resolution,
            rng_seed: grng_seed,
            'overwrite_files': self.sim_dict['overwrite_files'],
            'print_time': self.sim_dict['print_time'],
            'data_path': 'raw_data',
            # set presimulation-prefix already here to avoid empty files without
            # prefix
            'data_prefix': 'presim_'}

        if rng_seed == 'grng_seed': # in NEST 2
            kernel_dict.update({'rng_seeds': rng_seeds})

        nest.SetKernelStatus(kernel_dict)
        return

    def __create_neuronal_populations(self):
        """ Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from
        normal distributions dependent on the parameter ``V0_type``.

        The first and last neuron id of each population is written to file.
        """
        if nest.Rank() == 0:
            print('Creating neuronal populations.')

        self.pops = []
        for i in np.arange(self.net_dict['num_pops']):

            # random positions in 2D with periodic boundary conditions
            positions = nest.spatial.free(
                pos=nest.random.uniform(min=-self.net_dict['extent'] / 2.,
                                        max=self.net_dict['extent'] / 2.),
                edge_wrap=True,
                extent=[self.net_dict['extent'], self.net_dict['extent']])

            # cortical neuronal populations
            if i < self.net_dict['num_pops'] - 1:

                population = nest.Create(self.net_dict['neuron_model'],
                                         self.net_dict['num_neurons'][i],
                                         positions=positions)
                population.set(
                    tau_m=self.net_dict['neuron_params']['tau_m'],
                    tau_syn_ex=self.net_dict['neuron_params']['tau_syn'],
                    tau_syn_in=self.net_dict['neuron_params']['tau_syn'],
                    E_L=self.net_dict['neuron_params']['E_L'],
                    V_th=self.net_dict['neuron_params']['V_th'],
                    V_reset=self.net_dict['neuron_params']['V_reset'],
                    t_ref=self.net_dict['neuron_params']['t_ref'],
                    I_e=self.net_dict['DC_amp'][i])

                if self.net_dict['V0_type'] == 'optimized':
                    population.set(
                        V_m=nest.random.normal(
                            self.net_dict['neuron_params']['V0_mean']['optimized'][i],
                            self.net_dict['neuron_params']['V0_std']['optimized'][i]))
                elif self.net_dict['V0_type'] == 'original':
                    population.set(V_m=nest.random.normal(
                        self.net_dict['neuron_params']['V0_mean']['original'],
                        self.net_dict['neuron_params']['V0_std']['original']))
                else:
                    raise Exception(
                        'V0_type incorrect. ' +
                        'Valid options are "optimized" and "original".')

            # thalamic population
            else:
                population = nest.Create('parrot_neuron',
                                         self.net_dict['num_neurons'][-1],
                                         positions=positions)

            self.pops.append(population)

        # write node ids to file
        if nest.Rank() == 0:
            fn = os.path.join('raw_data', self.sim_dict['fname_nodeids'])
            with open(fn, 'w+') as f:
                for pop in self.pops:
                    f.write('{} {}\n'.format(pop[0].global_id,
                                             pop[-1].global_id))

        # Gather and write all positions to HDF5 file
        fn = os.path.join('raw_data', 'positions.h5')
        if nest.Rank() == 0:
            f = h5py.File(fn, 'w')
        for i, (label, pop) in enumerate(zip(self.net_dict['populations'],

                                             self.pops)):
            # extract layer nodes and positions on this RANK
            nodes = nest.GetLocalNodeCollection(pop)
            if len(nodes) > 1:
                pos = np.array(nest.GetPosition(nodes))
            elif len(nodes) == 1:
                pos = np.array(nest.GetPosition(nodes)).reshape((1, 2))
            else:
                pos = np.zeros((0, 2))

            # see ana_dict['read_nest_ascii_dtypes']['positions']
            # as ana_dict is not loaded here
            names = ['nodeid', 'x-position_mm', 'y-position_mm']
            formats = ['i4', 'f8', 'f8']

            # construct record arrau
            data = np.recarray((len(nodes), ), names=names, formats=formats)
            data['nodeid'] = nodes
            data['x-position_mm'] = pos[:, 0]
            data['y-position_mm'] = pos[:, 1]

            # Gather to RANK 0
            DATA = GathervRecordArray(data)

            # write
            if nest.Rank() == 0:
                f[label] = DATA

        if nest.Rank() == 0:
            f.close()

        return

    def __create_recording_devices(self):
        """ Creates one recording device of each kind per population.

        Only devices which are given in ``sim_dict['rec_dev']`` are created.
        The recorder label is equal to the respective name of the recording
        device.

        """
        if nest.Rank() == 0:
            print('Creating recording devices.')

        if 'spike_recorder' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('  Creating spike recorders.')

            sd_dict = {'record_to': 'memory'}
            self.spike_recorders = nest.Create('spike_recorder',
                                               n=self.net_dict['num_pops'],
                                               params=sd_dict)

            # cannot provide list of labels with params
            sd_labels = [
                'spike_recorder_' +
                pop for pop in self.net_dict['populations']]
            for i, sd in enumerate(self.spike_recorders):
                sd.label = sd_labels[i]

        if 'voltmeter' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('  Creating voltmeters.')
            vm_dict = {'interval': self.sim_dict['rec_V_int'],
                       'record_to': 'ascii',
                       'record_from': ['V_m']}

            self.voltmeters = nest.Create('voltmeter',
                                          n=self.net_dict['num_pops'] - 1,
                                          params=vm_dict)

            # cannot provide list of labels with params
            vm_labels = \
                ['voltmeter_' + pop for pop in self.net_dict['populations']]
            for i, vm in enumerate(self.voltmeters):
                vm.label = vm_labels[i]
        return

    def __create_poisson_bg_input(self):
        """ Creates the Poisson generators for ongoing background input if
        specified in ``network_params.py``.

        If ``poisson_input`` is ``False``, DC input is applied for compensation
        in ``create_neuronal_populations()``.

        """
        if nest.Rank() == 0:
            print('Creating Poisson generators for background input.')

        self.poisson_bg_input = nest.Create('poisson_generator',
                                            n=self.net_dict['num_pops'] - 1)
        self.poisson_bg_input.rate = \
            self.net_dict['bg_rate'] * self.net_dict['ext_indegrees']
        return

    def __create_thalamic_stim_input(self):
        """ Creates input for the thalamic neuronal population if specified in
        ``net_dict``.

        """
        if nest.Rank() == 0:
            print('Creating thalamic input for external stimulation.')

        # input to thalamic population
        if self.net_dict['thalamic_input'] == 'poisson':
            self.poisson_input_th = nest.Create('poisson_generator')
            self.poisson_input_th.set(
                rate=self.net_dict['th_rate'],
                start=self.net_dict['th_start'],
                stop=(
                    self.net_dict['th_start'] +
                    self.net_dict['th_duration']))

        elif self.net_dict['thalamic_input'] == 'pulses':
            # substract from pulse times the delay between pulse spike
            # generator and the thalamic population such that the first
            # thalamic pulse occurs exactly at th_pulse_start
            pulse_times = \
                np.arange(self.net_dict['th_pulse_start'],
                          self.sim_dict['t_presim'] + self.sim_dict['t_sim'],
                          self.net_dict['th_interval']) - \
                self.net_dict['th_delay_pulse_generator']

            # one spike generator at the center of the network
            self.spike_pulse_input_th = \
                nest.Create('spike_generator',
                            params={'spike_times': pulse_times},
                            positions=nest.spatial.grid(
                                shape=[1, 1],
                                edge_wrap=True))
        return

    def __create_dc_stim_input(self):
        """ Creates DC generators for external stimulation if specified
        in ``net_dict``.

        The final amplitude is the ``net_dict['dc_amp'] * net_dict['K_ext']``.

        """
        dc_amp_stim = self.net_dict['dc_amp'] * \
            self.net_dict['K_ext_' + self.net_dict['base_model']]

        if nest.Rank() == 0:
            print('Creating DC generators for external stimulation.')

        dc_dict = {'amplitude': dc_amp_stim,
                   'start': self.net_dict['dc_start'],
                   'stop': (self.net_dict['dc_start'] +
                            self.net_dict['dc_dur'])}
        self.dc_stim_input = nest.Create('dc_generator',
                                         # not for thalamus
                                         n=self.net_dict['num_pops'] - 1,
                                         params=dc_dict)
        return

    def __connect_neuronal_populations(self):
        """ Creates the recurrent connections between neuronal populations. """
        if nest.Rank() == 0:
            print('Connecting neuronal populations recurrently.')

        for i, target_pop in enumerate(
                self.pops[:-1]):  # thalamus is no target
            for j, source_pop in enumerate(self.pops):
                if self.net_dict['num_synapses'][i][j] >= 0.:

                    # TODO simplify these loops and conditions
                    # specify which connections exist
                    if self.net_dict['connect_method'] == 'fixedtotalnumber':
                        conn_dict_rec = {
                            'rule': 'fixed_total_number',
                            'N': self.net_dict['num_synapses'][i][j]}
                    elif self.net_dict['connect_method'] == 'fixedindegree':
                        conn_dict_rec = {
                            'rule': 'fixed_indegree',
                            'indegree': self.net_dict['indegrees'][i][j]}
                    elif self.net_dict['connect_method'] == 'fixedindegree_exp':
                        conn_dict_rec = {
                            'rule': 'fixed_indegree',
                            'indegree': self.net_dict['indegrees'][i][j],
                            'p': nest.spatial_distributions.exponential(
                                x=nest.spatial.distance,
                                beta=self.net_dict['beta'][i][j]),
                            'mask': {'circular': {
                                'radius': self.net_dict['extent'] / 2.}}}
                    elif self.net_dict['connect_method'] == 'distr_indegree_exp':
                        conn_dict_rec = {
                            'rule': 'pairwise_bernoulli',
                            'p': self.net_dict['p0'][i][j] *
                            nest.spatial_distributions.exponential(
                                    x=nest.spatial.distance,
                                    beta=self.net_dict['beta'][i][j]),
                            'mask': {'circular': {
                                'radius': self.net_dict['extent'] / 2.}}}
                    else:
                        raise Exception('connect_method is incorrect.')

                    # allow_multapses: True is ineffective for rule
                    # pairwise_bernoulli ('connect_method' == 'distr_indegree_exp')
                    conn_dict_rec.update({'allow_autapses': False,
                                          'allow_multapses': True})

                    # specify synapse parameters
                    if self.net_dict['weight_matrix_mean'][i][j] < 0:
                        w_min = np.NINF
                        w_max = 0.0
                    else:
                        w_min = 0.0
                        w_max = np.Inf

                    if self.net_dict['delay_type'] == 'normal':
                        delay_param = nest.random.normal(
                            mean=self.net_dict['delay_matrix_mean'][i][j],
                            std=(self.net_dict['delay_matrix_mean'][i][j] *
                                 self.net_dict['delay_rel_std']))
                    elif self.net_dict['delay_type'] == 'linear':
                        delay_param = (
                            (self.net_dict['delay_offset_matrix'][i][j] +
                             nest.spatial.distance /
                             self.net_dict['prop_speed_matrix'][i][j]) *
                            nest.random.normal(
                                mean=1.,
                                std=self.net_dict['delay_lin_rel_std']))

                    syn_dict = {
                        'synapse_model': 'static_synapse',
                        'weight': nest.math.redraw(
                            nest.random.normal(
                                mean=self.net_dict['weight_matrix_mean'][i][j],
                                std=abs(
                                    self.net_dict['weight_matrix_mean'][i][j] *
                                    self.net_dict['weight_rel_std'])),
                            min=w_min,
                            max=w_max),
                        'delay': nest.math.redraw(
                            delay_param,
                            min=self.sim_resolution,
                            max=np.Inf)}

                    # repeat_connect is 1 apart from rule pairwise_bernoulli
                    # ('connect_method' == 'distr_indegree_exp').
                    # note that for pairwise_bernoulli repeat_connect determines
                    # the maximum possible number of connections (multapses) for
                    # a pair of neurons
                    for repeat in np.arange(
                            self.net_dict['repeat_connect'][i][j]):
                        nest.Connect(
                            source_pop, target_pop,
                            conn_spec=conn_dict_rec,
                            syn_spec=syn_dict)
        return

    def __connect_recording_devices(self):
        """ Connects the recording devices to the mesocircuit."""
        if nest.Rank == 0:
            print('Connecting recording devices.')

        for i, target_pop in enumerate(self.pops):
            if 'spike_recorder' in self.sim_dict['rec_dev']:
                nest.Connect(target_pop, self.spike_recorders[i])
            if 'voltmeter' in self.sim_dict['rec_dev'] and i < len(
                    self.pops) - 1:
                nest.Connect(self.voltmeters[i], target_pop)
        return

    def __connect_poisson_bg_input(self):
        """ Connects the Poisson generators to the cortical populations."""
        if nest.Rank() == 0:
            print('Connecting Poisson generators for background input.')

        for i, target_pop in enumerate(self.pops[:-1]):  # not to thalamus
            conn_dict_poisson = {'rule': 'all_to_all'}

            syn_dict_poisson = {
                'synapse_model': 'static_synapse',
                'weight': self.net_dict['weight_ext'],
                'delay': self.net_dict['delay_poisson']}

            nest.Connect(
                self.poisson_bg_input[i], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson)
        return

    def __connect_thalamic_stim_input(self):
        """ Connects input to thalamic populations."""
        if nest.Rank() == 0:
            print('Connecting thalamic input.')

        # connect input to thalamic population
        if self.net_dict['thalamic_input'] == 'poisson':
            nest.Connect(self.poisson_input_th, self.pops[-1])
        elif self.net_dict['thalamic_input'] == 'pulses':
            conn_dict_pulse_th = {
                'rule': 'pairwise_bernoulli',
                'p': 1.0,
                'mask': {'circular': {'radius': self.net_dict['th_radius']}}}
            syn_dict_pulse_th = {
                'delay': self.net_dict['th_delay_pulse_generator']}

            nest.Connect(self.spike_pulse_input_th, self.pops[-1],
                         conn_spec=conn_dict_pulse_th,
                         syn_spec=syn_dict_pulse_th)
        return

    def __connect_dc_stim_input(self):
        """ Connects the DC generators to the neuronal populations. """

        if nest.Rank() == 0:
            print('Connecting DC generators.')

        for i, target_pop in enumerate(self.pops[:-1]):  # not to thalamus
            nest.Connect(self.dc_stim_input[i], target_pop)
        return
