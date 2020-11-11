"""PyNEST Mesocircuit: Network Class
------------------------------------

Main file of the mesocircuit defining the ``Network`` class with functions to
build and simulate the network.

"""

import os
import numpy as np
import nest

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
    stim_dict
        Dictionary containing all parameters specific to a potential stimulus
        (derived from: ``base_stimulus_params.py``)

    """

    def __init__(self, sim_dict, net_dict, stim_dict):
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict

        # check parameters and print information
        self.__check_parameters()

        # initialize the NEST kernel
        self.__setup_nest()

    
    def create(self):
        """ Creates all network nodes.

        Neuronal populations and recording and stimulating devices are created.

        """
        self.__create_neuronal_populations()
        if len(self.sim_dict['rec_dev']) > 0:
            self.__create_recording_devices()
        if self.net_dict['poisson_input']:
            self.__create_poisson_bg_input()
        if self.stim_dict['thalamic_input']:
            self.__create_thalamic_stim_input()
        if self.stim_dict['dc_input']:
            self.__create_dc_stim_input()


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
        if self.stim_dict['thalamic_input']:
            self.__connect_thalamic_stim_input()
        if self.stim_dict['dc_input']:
            self.__connect_dc_stim_input()

        nest.Prepare()
        nest.Cleanup()


    def presimulate(self, t_presim):
        """
        Wrapper for separate time measurement.
        kwargs is not passed on for not duplicating the measurement.
        """
        self.simulate(t_presim)

    def simulate(self, t_sim):
        """ Simulates the mesocircuit.

        Parameters
        ----------
        t_sim
            Simulation time (in ms).

        """
        if nest.Rank() == 0:
            print('Simulating {} ms.'.format(t_sim))

        nest.Simulate(t_sim)

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

    def __setup_nest(self):
        """ Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        The number of seeds for random number generation are computed based on
        the total number of virtual processes
        (number of MPI processes x number of threads per MPI process).
        """
        nest.ResetKernel()

        # set seeds for random number generation
        nest.SetKernelStatus(
            {'local_num_threads': self.sim_dict['local_num_threads']})
        N_vp = nest.GetKernelStatus('total_num_virtual_procs')

        master_seed = self.sim_dict['master_seed']
        grng_seed = master_seed + N_vp
        rng_seeds = (master_seed + N_vp + 1 + np.arange(N_vp)).tolist()

        if nest.Rank() == 0:
            print('Master seed: {} '.format(master_seed))
            print('  Total number of virtual processes: {}'.format(N_vp))
            print('  Global random number generator seed: {}'.format(grng_seed))
            print(
                '  Seeds for random number generators of virtual processes: ' +
                '{}'.format(rng_seeds))

        # pass parameters to NEST kernel
        self.sim_resolution = self.sim_dict['sim_resolution']
        kernel_dict = {
            'resolution': self.sim_resolution,
            'grng_seed': grng_seed,
            'rng_seeds': rng_seeds,
            'overwrite_files': self.sim_dict['overwrite_files'],
            'print_time': self.sim_dict['print_time']}
        nest.SetKernelStatus(kernel_dict)

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
                pos=nest.random.uniform(min=-self.net_dict['extent']/2.,
                                        max=self.net_dict['extent']/2.),
                edge_wrap=True,
                num_dimensions=2)

            population = nest.Create(self.net_dict['neuron_model'],
                                     self.net_dict['num_neurons'][i],
                                     positions=positions)
            population.set(
                tau_syn_ex=self.net_dict['neuron_params']['tau_syn'],
                tau_syn_in=self.net_dict['neuron_params']['tau_syn'],
                E_L=self.net_dict['neuron_params']['E_L'],
                V_th=self.net_dict['neuron_params']['V_th'],
                V_reset=self.net_dict['neuron_params']['V_reset'],
                t_ref=self.net_dict['neuron_params']['t_ref'],
                I_e=self.net_dict['DC_amp'][i])

            if self.net_dict['V0_type'] == 'optimized':
                population.set(V_m=nest.random.normal(
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

            self.pops.append(population)

        # write node ids to file
        if nest.Rank() == 0:
            fn = os.path.join(self.sim_dict['path_raw_data'],
                              self.sim_dict['fname_nodeids'])
            with open(fn, 'w+') as f:
                for pop in self.pops:
                    f.write('{} {}\n'.format(pop[0].global_id,
                                             pop[-1].global_id))

        # write MPI-local positions to file
        # rank is automatically appended to file name
        for i,pop in enumerate(self.pops):
            fn = os.path.join(
                self.sim_dict['path_raw_data'],
                'positions_' + self.net_dict['populations'][i] + '.dat')
            nest.DumpLayerNodes(pop, fn)


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

            sd_dict = {'record_to': 'ascii'}
            self.spike_recorders = nest.Create('spike_recorder',
                                               n=self.net_dict['num_pops'],
                                               params=sd_dict)

            # cannot provide list of labels with params
            sd_labels = [os.path.join(self.sim_dict['path_raw_data'],
                                      'spike_recorder_' + pop) \
                         for pop in self.net_dict['populations']]
            for i,sd in enumerate(self.spike_recorders):
                sd.label = sd_labels[i]

        if 'voltmeter' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('  Creating voltmeters.')
            vm_dict = {'interval': self.sim_dict['rec_V_int'],
                       'record_to': 'ascii',
                       'record_from': ['V_m']}
                                             
            self.voltmeters = nest.Create('voltmeter',
                                          n=self.net_dict['num_pops'],
                                          params=vm_dict)

            # cannot provide list of labels with params
            vm_labels = [os.path.join(self.sim_dict['path_raw_data'],
                                      'voltmeter_' + pop) \
                         for pop in self.net_dict['populations']]
            for i,vm in enumerate(self.voltmeters):
                vm.label = vm_labels[i]


    def __create_poisson_bg_input(self):
        """ Creates the Poisson generators for ongoing background input if
        specified in ``network_params.py``.

        If ``poisson_input`` is ``False``, DC input is applied for compensation
        in ``create_neuronal_populations()``.

        """
        if nest.Rank() == 0:
            print('Creating Poisson generators for background input.')

        self.poisson_bg_input = nest.Create('poisson_generator',
                                            n=self.net_dict['num_pops'])
        self.poisson_bg_input.rate = \
            self.net_dict['bg_rate'] * self.net_dict['ext_indegrees']

    def __create_thalamic_stim_input(self):
        """ Creates the thalamic neuronal population if specified in
        ``stim_dict``.

        Thalamic neurons are of type ``parrot_neuron`` and receive input from a
        Poisson generator.
        Note that the number of thalamic neurons is not scaled with
        ``N_scaling``.

        """
        if nest.Rank() == 0:
            print('Creating thalamic input for external stimulation.')

        self.thalamic_population = nest.Create(
            'parrot_neuron', n=self.stim_dict['num_th_neurons'])

        self.poisson_th = nest.Create('poisson_generator')
        self.poisson_th.set(
            rate=self.stim_dict['th_rate'],
            start=self.stim_dict['th_start'],
            stop=(self.stim_dict['th_start'] + self.stim_dict['th_duration']))

    def __create_dc_stim_input(self):
        """ Creates DC generators for external stimulation if specified
        in ``stim_dict``.

        The final amplitude is the ``stim_dict['dc_amp'] * net_dict['K_ext']``.

        """
        dc_amp_stim = self.stim_dict['dc_amp'] * self.net_dict['K_ext']

        if nest.Rank() == 0:
            print('Creating DC generators for external stimulation.')

        dc_dict = {'amplitude': dc_amp_stim,
                   'start': self.stim_dict['dc_start'],
                   'stop': (self.stim_dict['dc_start'] +
                            self.stim_dict['dc_dur'])}
        self.dc_stim_input = nest.Create('dc_generator', n=self.net_dict['num_pops'],
                                         params=dc_dict)

    def __connect_neuronal_populations(self):
        """ Creates the recurrent connections between neuronal populations. """
        if nest.Rank() == 0:
            print('Connecting neuronal populations recurrently.')

        for i, target_pop in enumerate(self.pops):
            for j, source_pop in enumerate(self.pops):
                if self.net_dict['num_synapses'][i][j] >= 0.:

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
                    else:
                        raise Exception('connect_method is incorrect.')

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
                                std=abs(self.net_dict['weight_matrix_mean'][i][j] *
                                        self.net_dict['weight_rel_std'])),
                            min=w_min,
                            max=w_max),
                        'delay': nest.math.redraw(
                            delay_param,
                            min=self.sim_resolution,
                            max=np.Inf)}

                    nest.Connect(
                        source_pop, target_pop,
                        conn_spec=conn_dict_rec,
                        syn_spec=syn_dict)


    def __connect_recording_devices(self):
        """ Connects the recording devices to the mesocircuit."""
        if nest.Rank == 0:
            print('Connecting recording devices.')

        for i, target_pop in enumerate(self.pops):
            if 'spike_recorder' in self.sim_dict['rec_dev']:
                nest.Connect(target_pop, self.spike_recorders[i])
            if 'voltmeter' in self.sim_dict['rec_dev']:
                nest.Connect(self.voltmeters[i], target_pop)

    def __connect_poisson_bg_input(self):
        """ Connects the Poisson generators to the mesocircuit."""
        if nest.Rank() == 0:
            print('Connecting Poisson generators for background input.')

        for i, target_pop in enumerate(self.pops):
            conn_dict_poisson = {'rule': 'all_to_all'}

            syn_dict_poisson = {
                'synapse_model': 'static_synapse',
                'weight': self.net_dict['weight_ext'],
                'delay': self.net_dict['delay_poisson']}

            nest.Connect(
                self.poisson_bg_input[i], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson)

    def __connect_thalamic_stim_input(self):
        """ Connects the thalamic input to the neuronal populations."""
        if nest.Rank() == 0:
            print('Connecting thalamic input.')

        # connect Poisson input to thalamic population
        nest.Connect(self.poisson_th, self.thalamic_population)

        # connect thalamic population to neuronal populations
        for i, target_pop in enumerate(self.pops):
            conn_dict_th = {
                'rule': 'fixed_total_number',
                'N': self.stim_dict['num_th_synapses'][i]}

            syn_dict_th = {
                'weight': nest.math.redraw(
                    nest.random.normal(
                        mean=self.net_dict['weight_th'],
                        std=self.net_dict['weight_th'] * self.net_dict['weight_rel_std']),
                    min=0.0,
                    max=np.Inf),
                'delay': nest.math.redraw(
                    nest.random.normal(
                        mean=self.stim_dict['delay_th_mean'],
                        std=(self.stim_dict['delay_th_mean'] *
                             self.stim_dict['delay_th_rel_std'])),
                    min=self.sim_resolution,
                    max=np.Inf)}

            nest.Connect(
                self.thalamic_population, target_pop,
                conn_spec=conn_dict_th, syn_spec=syn_dict_th)

    def __connect_dc_stim_input(self):
        """ Connects the DC generators to the neuronal populations. """

        if nest.Rank() == 0:
            print('Connecting DC generators.')

        for i, target_pop in enumerate(self.pops):
            nest.Connect(self.dc_stim_input[i], target_pop)
