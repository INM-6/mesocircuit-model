#!/usr/bin/env python
'''
Hybrid LFP scheme example script, applying the methodology with a model
implementation similar to:

Nicolas Brunel. "Dynamics of Sparsely Connected Networks of Excitatory and
Inhibitory Spiking Neurons". J Comput Neurosci, May 2000, Volume 8,
Issue 3, pp 183-208

But the network is implemented with spatial connectivity, i.e., the neurons
are assigned positions and distance-dependent connectivity in terms of
cell-cell connectivity and transmission delays.

Synopsis of the main simulation procedure:
1. Loading of parameterset
    a. network parameters
    b. parameters for hybrid scheme
2. Set up file destinations for different simulation output
3. network simulation
    a. execute network simulation using NEST (www.nest-initiative.org)
    b. merge network output (spikes, currents, voltages)
4. Create a object-representation that uses sqlite3 of all the spiking output
5. Iterate over post-synaptic populations:
    a. Create Population object with appropriate parameters for
       each specific population
    b. Run all computations for populations
    c. Postprocess simulation output of all cells in population
6. Postprocess all cell- and population-specific output data
7. Create a tarball for all non-redundant simulation output

The full simulation can be evoked by issuing a mpirun call, such as
mpirun -np 4 python example_brunel.py

Not recommended, but running it serially should also work, e.g., calling
python example_brunel.py


Given the size of the network and demands for the multi-compartment LFP-
predictions using the present scheme, running the model on nothing but a large-
scale compute facility is strongly discouraged.
'''
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
from time import time
import neuron  # needs to be imported before MPI
from hybridLFPy import PostProcess, CachedTopoNetwork, TopoPopulation
# setup_file_dest,
import pickle
from periodiclfp import PeriodicLFP
from lfp_parameters import get_parameters
from mpi4py import MPI


#################################################
# matplotlib settings                           #
#################################################
plt.close('all')
plt.rcParams.update({'figure.figsize': [10.0, 8.0]})


# set some seed values
SEED = 12345678
SIMULATIONSEED = 12345678
np.random.seed(SEED)


#################################################
# Initialization of MPI stuff                   #
#################################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


# if True, execute full model. If False, do only the plotting.
# Simulation results must exist.
PROPERRUN = True


# check if mod file for synapse model specified in alphaisyn.mod is loaded
if not hasattr(neuron.h, 'ExpSynI'):
    if RANK == 0:
        os.system('nrnivmodl')
    COMM.Barrier()
    neuron.load_mechanisms('.')


##########################################################################
# HYBRID SCHEME PARAMETERS
##########################################################################

path_parameters = sys.argv[1]

dics = []
for dic in ['sim_dict', 'net_dict', 'stim_dict']:
    with open(os.path.join(path_parameters, dic + '.pkl'), 'rb') as f:
        dics.append(pickle.load(f))
sim_dict, net_dict, _ = dics  # stim dict not neeeded here?


##########################################################################
# MAIN simulation procedure
##########################################################################

# if PROPERRUN:
#    # set up the file destination, removing old results by default
#    setup_file_dest(PS, clearDestination=False)
path_lfp_data = os.path.join(sim_dict['data_path'], 'lfp',
                             os.path.split(path_parameters)[-1])
# path_lfp_cells_data = os.path.join(path_lfp_data, 'cells')
# path_lfp_populations_data = os.path.join(path_lfp_data, 'populations')
# path_lfp_figures = os.path.join(path_lfp_data, 'figures')
if RANK == 0:
    if not os.path.isdir(os.path.split(path_lfp_data)[0]):
        os.mkdir(os.path.split(path_lfp_data)[0])
    if not os.path.isdir(path_lfp_data):
        os.mkdir(path_lfp_data)
    '''
    if not os.path.isdir(path_lfp_cells_data):
        os.mkdir(path_lfp_cells_data)
    if not os.path.isdir(path_lfp_populations_data):
        os.mkdir(path_lfp_populations_data)
    if not os.path.isdir(path_lfp_figures):
        os.mkdir(path_lfp_figures)
    '''

# wait for operation to finish
COMM.Barrier()

# get ParameterSet object instance with all required parameters for LFPs etc.
PS = get_parameters(path_lfp_data=path_lfp_data,
                    sim_dict=sim_dict,
                    net_dict=net_dict)

if RANK == 0:
    simstats = open(os.path.join(PS.savefolder, 'simstats.dat'), 'w')


# tic toc
tic = time()
ticc = tic

# Create an object representation containing the spiking activity of the
# network simulation output that uses sqlite3.
networkSim = CachedTopoNetwork(**PS.network_params)


tocc = time()
if RANK == 0:
    simstats.write('CachedNetwork {}\n'.format(tocc - ticc))

toc = time() - tic
print(('NEST simulation and gdf file processing done in  %.3f seconds' % toc))


##############################################################################
# Create predictor for extracellular potentials that utilize periodic
# boundary conditions in 2D, similar to network connectivity.
# Spatial layout similar to Utah-array
##############################################################################
# Set up LFPykit measurement probes for LFPs and CSDs
if PROPERRUN:
    probes = []
    probes.append(PeriodicLFP(cell=None, **PS.electrodeParams))

##############################################################################
# Create multicompartment neuron populations for LFP predictions
##############################################################################
if PROPERRUN:
    # iterate over each cell type, and create populationulation object
    for i, y in enumerate(PS.y):
        # create population:
        ticc = time()
        pop = TopoPopulation(
            cellParams=PS.cellParams[y],
            rand_rot_axis=PS.rand_rot_axis[y],
            simulationParams=PS.simulationParams,
            populationParams=PS.populationParams[y],
            y=y,
            layerBoundaries=PS.layerBoundaries,
            probes=probes,
            savelist=PS.savelist,
            savefolder=PS.savefolder,
            dt_output=PS.dt_output,
            POPULATIONSEED=SIMULATIONSEED + i,
            X=PS.X,
            networkSim=networkSim,
            k_yXL=PS.k_yXL[y],
            synParams=PS.synParams[y],
            synDelayLoc=PS.synDelayLoc[y],
            synDelayScale=PS.synDelayScale[y],
            J_yX=PS.J_yX[y],
            tau_yX=PS.tau_yX[y],
            # TopoPopulation kwargs
            topology_connections=PS.topology_connections,
        )

        tocc = time()
        if RANK == 0:
            simstats.write('Population_{} {}\n'.format(y, tocc - ticc))

        # run population simulation and collect the data
        ticc = time()
        pop.run()
        tocc = time()
        if RANK == 0:
            simstats.write('run_{} {}\n'.format(y, tocc - ticc))

        ticc = time()
        pop.collect_data()
        tocc = time()

        if RANK == 0:
            simstats.write('collect_{} {}\n'.format(y, tocc - ticc))

        # object no longer needed
        del pop

##############################################################################
# Postprocess the simulation output (sum up contributions by each cell type)
##############################################################################
# reset seed, but output should be deterministic from now on
np.random.seed(SIMULATIONSEED)

if PROPERRUN:
    ticc = time()
    # do some postprocessing on the collected data, i.e., superposition
    # of population LFPs, CSDs etc
    postproc = PostProcess(y=PS.y,
                           dt_output=PS.dt_output,
                           savefolder=PS.savefolder,
                           mapping_Yy=PS.mapping_Yy,
                           savelist=PS.savelist,
                           cells_subfolder=os.path.split(PS.cells_path)[-1],
                           populations_subfolder=os.path.split(
                               PS.populations_path)[-1],
                           figures_subfolder=PS.figures_subfolder,
                           )

    # run through the procedure
    postproc.run()

    # create tar-archive with output for plotting, ssh-ing etc.
    # postproc.create_tar_archive()

    tocc = time()
    if RANK == 0:
        simstats.write('postprocess {}\n'.format(tocc - ticc))
        simstats.close()

COMM.Barrier()

# tic toc
print(('Execution time: %.3f seconds' % (time() - tic)))


'''
##########################################################################
# Create animations from simulation output
##########################################################################

# if RANK == 0:
#    network_lfp_activity_nolabels_animation
#    fig = network_lfp_activity_nolabels_animation(PS, PSET, networkSim,
#        T=(100, 200), save_anim=True)

if RANK == 0 and not PROPERRUN:
    fig = network_activity_animation(
        PS, PSET, networkSim, T=(
            100, 300), N_X=(
            PS.N_X / 16. / PSET.density).astype(int), save_anim=True)
    plt.close(fig)

if RANK == 0 and not PROPERRUN:
    fig = lfp_activity_animation(PS, PSET, T=(100, 300), save_anim=True)
    plt.close(fig)

if RANK == 0 and not PROPERRUN:
    fig = network_lfp_activity_animation(
        PS, PSET, networkSim, T=(
            100, 300), N_X=(
            PS.N_X / 16. / PSET.density).astype(int), save_anim=True)
    plt.close(fig)

COMM.Barrier()
'''
