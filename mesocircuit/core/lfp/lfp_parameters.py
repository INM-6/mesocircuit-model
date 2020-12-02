#!/usr/bin/env python
'''derived parameters for hybridLFPy forward-model predictions'''
import os
import numpy as np
from parameters import ParameterSet
import json


def flattenlist(lst):
    return sum(sum(lst, []), [])


def get_L_yXL(fname, y, x_in_X, L):
    '''
    compute the layer specificity, defined as:
    ::

        L_yXL = k_yXL / k_yX
    '''
    def _get_L_yXL_per_yXL(fname, x_in_X, X_index,
                           y, layer):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()

        # Get number of synapses
        if layer in [str(key)
                     for key in list(data['data'][y]['syn_dict'].keys())]:
            # init variables
            k_yXL = 0
            k_yX = 0

            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][layer][x] / 100.
                k_yL = data['data'][y]['syn_dict'][layer][
                    'number of synapses per neuron']
                k_yXL += p_yxL * k_yL

            for ll in [str(key)
                       for key in list(data['data'][y]['syn_dict'].keys())]:
                for x in x_in_X[X_index]:
                    p_yxL = data['data'][y]['syn_dict'][ll][x] / 100.
                    k_yL = data['data'][y]['syn_dict'][ll][
                        'number of synapses per neuron']
                    k_yX += p_yxL * k_yL

            if k_yXL != 0.:
                return k_yXL / k_yX
            else:
                return 0.
        else:
            return 0.

    # init dict
    L_yXL = {}

    # iterate over postsynaptic cell types
    for y_value in y:
        # container
        data = np.zeros((len(L), len(x_in_X)))
        # iterate over lamina
        for i, Li in enumerate(L):
            # iterate over presynapse population inds
            for j in range(len(x_in_X)):
                data[i][j] = _get_L_yXL_per_yXL(fname, x_in_X,
                                                X_index=j,
                                                y=y_value,
                                                layer=Li)
        L_yXL[y_value] = data

    return L_yXL


def get_T_yX(fname, y, y_in_Y, x_in_X, F_y):
    '''
    compute the cell type specificity, defined as:
    ::

        T_yX = K_yX / K_YX
            = F_y * k_yX / sum_y(F_y*k_yX)

    '''
    def _get_k_yX_mul_F_y(y, y_index, X_index):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()

        # init variables
        k_yX = 0.

        for ll in [str(key)
                   for key in list(data['data'][y]['syn_dict'].keys())]:
            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][ll][x] / 100.
                k_yL = data['data'][y]['syn_dict'][ll][
                    'number of synapses per neuron']
                k_yX += p_yxL * k_yL

        return k_yX * F_y[y_index]

    # container
    T_yX = np.zeros((len(y), len(x_in_X)))

    # iterate over postsynaptic cell types
    for i, y_value in enumerate(y):
        # iterate over presynapse population inds
        for j in range(len(x_in_X)):
            k_yX_mul_F_y = 0
            for k, yy in enumerate(sum(y_in_Y, [])):
                if y_value in yy:
                    for yy_value in yy:
                        ii = np.where(np.array(y) == yy_value)[0][0]
                        k_yX_mul_F_y += _get_k_yX_mul_F_y(yy_value, ii, j)

            if k_yX_mul_F_y != 0:
                T_yX[i, j] = _get_k_yX_mul_F_y(y_value, i, j) / k_yX_mul_F_y

    return T_yX


class ParamsLFP(ParameterSet):
    '''extend parameters.ParameterSet class with some methods'''

    def __init__(self, initializer):
        super().__init__(initializer)

        # cell types
        self.y = flattenlist(self.y_in_Y)

        # Frequency of occurrence of each cell type (F_y); 1-d array
        self.F_y = self._get_F_y(self.y)

        # Relative frequency of occurrence of each cell type within its
        # population (F_{y,Y})
        self.F_yY = [[self._get_F_y(y) for y in Y] for Y in self.y_in_Y]

        # assess depth of each 16 subpopulation
        self.depths = self._calcDepths()

    def _calcDepths(self):
        '''
        return the cortical depth of each subpopulation
        '''
        depths = self.layerBoundaries.mean(axis=1)[1:]
        depth_y = []
        for y in self.y:
            if y in ['p23', 'b23', 'nb23']:
                depth_y = np.r_[depth_y, depths[0]]
            elif y in ['p4', 'ss4(L23)', 'ss4(L4)', 'b4', 'nb4']:
                depth_y = np.r_[depth_y, depths[1]]
            elif y in ['p5(L23)', 'p5(L56)', 'b5', 'nb5']:
                depth_y = np.r_[depth_y, depths[2]]
            elif y in ['p6(L4)', 'p6(L56)', 'b6', 'nb6']:
                depth_y = np.r_[depth_y, depths[3]]
            else:
                raise NotImplementedError
        return depth_y

    def _getCellParams(self):
        '''
        Return dict with parameters for each population.
        The main operation is filling in cell type specific morphology
        '''
        # cell type specific parameters going into LFPy.Cell
        yCellParams = {}
        for layer, morpho, _, _ in self.y_zip_list:
            yCellParams.update({layer: self.sharedCellParams.copy()})
            yCellParams[layer].update({
                'morphology': os.path.join(self.PATH_m_y, morpho),
            })
        return yCellParams

    def _get_F_y(self, y):
        '''
        Extract frequency of occurrences of those cell types that are modeled.
        The data set contains cell types that are not modeled (TCs etc.)
        The returned percentages are renormalized onto modeled cell-types,
        i.e. they sum up to 1
        '''
        # Load data from json dictionary
        f = open(self.connectivity_table, 'r')
        data = json.load(f)
        f.close()

        occurr = []
        for cell_type in y:
            occurr += [data['data'][cell_type]['occurrence']]
        return list(np.array(occurr) / np.sum(occurr))


def get_parameters(path_lfp_data=None, sim_dict=None, net_dict=None):
    '''
    get parameters.ParameterSet like object with all parameters required
    for LFP predictions using hybridLFPy

    Parameters
    ----------
    path_lfp_data: path
        simuation output folder
    sim_dict: dict
        global simulation parameter dictionary
    net_dict: dict
        network simulation parameter dictionary
    '''
    # get source directory
    file_prefix = os.path.split(__file__)[0]

    # initialize parameter object
    PS = ParamsLFP(dict(
        savefolder=path_lfp_data,
        # derived parameters for CachedTopoNetwork instance
        network_params=dict(
            simtime=sim_dict['t_presim'] + sim_dict['t_sim'],
            dt=sim_dict['sim_resolution'],
            spike_output_path=sim_dict['path_processed_data'],
            label=sim_dict['rec_dev'][0],
            ext='dat',
            GIDs=dict(zip(net_dict['populations'],
                          [[0, N_X] for N_X in net_dict['num_neurons']])),
            X=net_dict['populations'],
            label_positions='positions',
            skiprows=3,
        ),
        cells_path=os.path.join(path_lfp_data, 'cells'),
        populations_path=os.path.join(path_lfp_data, 'populations'),
        figures_subfolder=os.path.join(path_lfp_data, 'figures'),
        dt_output=1.,
        electrodeParams=dict(
            # contact locations:
            x=np.meshgrid(np.linspace(-1800, 1800, 10),
                          np.linspace(-1800, 1800, 10))[0].flatten(),
            y=np.meshgrid(np.linspace(-1800, 1800, 10),
                          np.linspace(-1800, 1800, 10))[1].flatten(),
            z=[-400. for x in range(100)],
            # extracellular conductivity:
            sigma=0.3,
            # contact surface normals, radius, n-point averaging
            N=[[0, 0, 1]] * 100,
            r=5,
            n=20,
            seedvalue=None,
            # dendrite line sources, soma as sphere source (Linden2014)
            method='root_as_point',
        ),
        X=net_dict['populations'].tolist(),  # add TC!!!
        Y=net_dict['populations'].tolist(),
        N_X=net_dict['num_neurons'],
        N_Y=net_dict['num_neurons'].tolist(),
        y_in_Y=[
            [['p23'], ['b23', 'nb23']],
            [['p4', 'ss4(L23)', 'ss4(L4)'], ['b4', 'nb4']],
            [['p5(L23)', 'p5(L56)'], ['b5', 'nb5']],
            [['p6(L4)', 'p6(L56)'], ['b6', 'nb6']]],
        layerBoundaries=np.array([[0.0, -81.6],
                                  [-81.6, -587.1],
                                  [-587.1, -922.2],
                                  [-922.2, -1170.0],
                                  [-1170.0, -1491.7]]),
        connectivity_table=os.path.join(file_prefix,
                                        'binzegger_connectivity_table.json'),
        sharedCellParams={
            'cm': 1.0,
            'Ra': 150,
            'v_init': net_dict['neuron_params']['E_L'],
            'passive': True,
            'passive_parameters': {
                # assumes cm=1:
                'g_pas': 1. / (net_dict['neuron_params']['tau_m'] * 1E3),
                'e_pas': net_dict['neuron_params']['E_L']},
            'nsegs_method': 'lambda_f',
            'lambda_f': 100,
            'dt': sim_dict['sim_resolution'],
            'tstart': 0,
            'tstop': sim_dict['t_presim'] + sim_dict['t_sim'],
            'verbose': False,
        }
    ))

    # map network populations Y onto cell type y
    PS.mapping_Yy = list(zip(
        ['L23E', 'L23I', 'L23I',
         'L4E', 'L4E', 'L4E', 'L4I', 'L4I',
         'L5E', 'L5E', 'L5I', 'L5I',
         'L6E', 'L6E', 'L6I', 'L6I'],
        PS.y))

    # define morphology file paths
    testing = True  # if True, use ball-and-stick type morphologies
    if testing:
        PS.PATH_m_y = os.path.join(file_prefix, 'morphologies')
        PS.m_y = [Y + '_' + y + '.hoc' for Y, y in PS.mapping_Yy]
    else:
        PS.PATH_m_y = os.path.join(file_prefix, 'morphologies', 'Hagen2016')
        PS.m_y = [
            'L23E_oi24rpy1.hoc',
            'L23I_oi38lbc1.hoc',
            'L23I_oi38lbc1.hoc',

            'L4E_53rpy1.hoc',
            'L4E_j7_L4stellate.hoc',
            'L4E_j7_L4stellate.hoc',
            'L4I_oi26rbc1.hoc',
            'L4I_oi26rbc1.hoc',

            'L5E_oi15rpy4.hoc',
            'L5E_j4a.hoc',
            'L5I_oi15rbc1.hoc',
            'L5I_oi15rbc1.hoc',

            'L6E_51-2a.CNG.hoc',
            'L6E_oi15rpy4.hoc',
            'L6I_oi15rbc1.hoc',
            'L6I_oi15rbc1.hoc',
        ]

    # Number of neurons of each cell type (N_y); 1-d array
    PS.N_y = np.round(
        np.array([PS.N_Y[layer * 2 + pop] * PS.F_yY[layer][pop][k]
                  for layer, y_in_Y in enumerate(PS.y_in_Y)
                  for pop, cell_types in enumerate(y_in_Y)
                  for k, _ in enumerate(cell_types)])).astype(int)
    # PS.N_y *= PSET.density
    # PS.N_y = PS.N_y.astype(int)

    # make a nice structure with data for each subpopulation
    PS.y_zip_list = list(zip(PS.y, PS.m_y, PS.depths, PS.N_y))

    # layer specific LFPy.Cell-parameters as nested dictionary
    PS.cellParams = PS._getCellParams()

    # set the axis of which each cell type y is randomly rotated,
    # SS types and INs are rotated around both x- and z-axis
    # in the population class, while P-types are
    # only rotated around the z-axis
    PS.rand_rot_axis = {}
    for y, _, _, _ in PS.y_zip_list:
        # identify pyramidal cell populations:
        if y.rfind('p') >= 0:
            PS.rand_rot_axis.update({y: ['z']})
        else:
            PS.rand_rot_axis.update({y: ['x', 'z']})

    # additional simulation kwargs, see LFPy.Cell.simulate() docstring
    PS.simulationParams = {}

    # a dict setting the number of cells N_y and geometry
    # of cell type population y
    PS.populationParams = {}
    for y, _, depth, N_y in PS.y_zip_list:
        PS.populationParams.update({
            y: {
                'number': N_y,
                'z_min': depth - 25,
                'z_max': depth + 25,
            }
        })

    # explicitly set the first neuron position index for each cell type y by
    # distributing from the proper network neuron positions
    Y0 = None
    count = 0
    for i, (Y, y) in enumerate(PS.mapping_Yy):
        if Y != Y0:
            count = 0
        PS.populationParams[y].update(dict(position_index_in_Y=[Y, count]))
        count += PS.N_y[i]
        Y0 = Y

    # LFPy.Cell instance attributes which will be saved
    PS.savelist = []

    # need presynaptic cell type to population mapping
    # PS.x_in_X = [['TCs', 'TCn']] + sum(self.y_in_Y, [])
    PS.x_in_X = sum(PS.y_in_Y, [])

    ############################################
    # Compute layer specificity (L_yXL) etc.
    ############################################

    # concatenate number of synapses of TC and cortical populations
    # K_YX = np.c_[np.array(PSET.full_num_synapses_th),
    #             np.array(PSET.full_num_synapses)].astype(float)
    K_YX = net_dict['num_synapses']

    # Scale the number of synapses according to network density parameter
    # K_YX *= PSET.density
    # K_YX = K_YX.astype(int)

    # spatial connection probabilites on each subpopulation
    # Each key must correspond to a subpopulation like 'L23E' used everywhere
    # else,
    # each array maps thalamic and intracortical connections.
    # First column is thalamic connections, and the rest intracortical,
    # ordered like 'L23E', 'L23I' etc., first row is normalised probability of
    # connection withing L1, L2, etc.;
    PS.L_yXL = get_L_yXL(fname=PS.connectivity_table,
                         y=PS.y,
                         x_in_X=PS.x_in_X,
                         L=['1', '23', '4', '5', '6'])

    # compute the cell type specificity
    PS.T_yX = get_T_yX(fname=PS.connectivity_table,
                       y=PS.y,
                       y_in_Y=PS.y_in_Y,
                       x_in_X=PS.x_in_X,
                       F_y=PS.F_y)

    # assess relative distribution of synapses for a given celltype
    PS.K_yXL = {}
    for i, (Y, y) in enumerate(PS.mapping_Yy):
        # fill in K_yXL (layer specific connectivity)
        PS.K_yXL[y] = (PS.T_yX[i, ] *
                       K_YX[np.array(PS.Y) == Y, ] *
                       PS.L_yXL[y]).astype(int)

    # number of incoming connections per cell type per layer per cell
    PS.k_yXL = {}
    for y, N_y in zip(PS.y, PS.N_y):
        PS.k_yXL.update({y: (PS.K_yXL[y] / N_y).astype(int)})

    # Set up cell type specific synapse parameters in terms of synapse model
    # and synapse locations
    PS.synParams = {}
    for y in PS.y:
        if y.rfind('p') >= 0:
            # pyramidal types have apical dendrite sections
            section = ['apic', 'dend']
        else:
            # other cell types do not
            section = ['dend']

        PS.synParams.update({
            y: {
                'syntype': 'ExpSynI',  # current based exponential synapse
                'section': section,
                # 'tau' : PSET.model_params["/tau_syn_ex"],
            },
        })

    # set up delays, here using fixed delays of network unless value is None
    PS.synDelayLoc = {y: [None for X in PS.X] for y in PS.y}

    # distribution of delays added on top of per connection delay using either
    # fixed or linear distance-dependent delays
    PS.synDelayScale = {y: [None for X in PS.X] for y in PS.y}
    # TODO@JSE: NOT SURE WHAT delay* entry in "net_dict" this corresponds to

    # PSC amplitues
    PS.J_YX = net_dict['weight_matrix_mean'] * 1e-3  # pA -> nA unit conversion
    PS.J_yX = {}
    for Y, y in PS.mapping_Yy:
        [i] = np.where(np.array(PS.Y) == Y)[0]
        PS.J_yX.update({y: PS.J_YX[i, ]})

    # set up dictionary of synapse time constants specific to each
    # postsynaptic cell type and presynaptic population, such that
    # excitatory and inhibitory time constants can be varied.
    PS.tau_yX = {}
    for y in PS.y:
        PS.tau_yX.update({
            y: [net_dict['neuron_params']['tau_syn'] for X in PS.X]
        })

    # set parameters for topology connections with spatial parameters
    # converted to units of mum from mm.
    PS.topology_connections = {}
    for i, X in enumerate(PS.X):
        PS.topology_connections[X] = {}
        for Y, y in PS.mapping_Yy:
            [j] = np.where(np.array(PS.Y) == Y)[0]

            PS.topology_connections[X][y] = dict(
                extent=[net_dict['extent'] * 1E3,
                        net_dict['extent'] * 1E3],
                edge_wrap=True,
                allow_autapses=True,
                mask=dict(
                    circular=dict(radius=net_dict['extent'] * 1E3 / 2)
                ),
                delays=dict(
                    linear=dict(
                        c=net_dict['delay_offset_matrix'][j, i],
                        # ms/mm -> ms/mum conversion):
                        a=net_dict['prop_speed_matrix'][j, i] * 1E-3
                    )
                )
            )

            # we may have different kernel types for different connections
            if net_dict['connect_method'] == 'fixedindegree_gaussian':
                # not implemented yet in network simulation????
                PS.topology_connections[X][y].update({
                    'kernel': {
                        'gaussian': dict(
                            p_center=1.,
                            mean=net_dict['std'][j, i] * 1E3,
                            c=0.,
                            # mm -> mum unit conversion
                            sigma=net_dict['std'][j, i] * 1E3
                        )
                    }
                })
            elif net_dict['connect_method'] == 'fixedindegree_exp':
                PS.topology_connections[X][y].update({
                    'kernel': {
                        'exponential': dict(
                            a=1,
                            c=0,
                            # mm -> mum unit conversion:
                            tau=net_dict['beta'][j, i] * 1E3
                        )
                    }
                })
            elif net_dict['connect_method'] == 'random':
                PS.topology_connections[X][y].update({
                    'kernel': 'random'
                })
            else:
                mssg = 'connect_method {} not implemented'.format(
                    net_dict['connect_method'])
                raise NotImplementedError(mssg)

    return PS


if __name__ == '__main__':
    PS = get_parameters()
    print(PS)
