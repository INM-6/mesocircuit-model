#!/usr/bin/env python
'''
LFP simulation script for 4x4 mm2 network model

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
import sys
from mpi4py import MPI
from lfpykit import CurrentDipoleMoment, VolumetricCurrentSourceDensity
import mesocircuit
import mesocircuit.mesocircuit_framework as mesoframe
from mesocircuit.lfp.lfp_parameters import get_parameters
from mesocircuit.lfp.periodiclfp import PeriodicLFP
from hybridLFPy import CachedTopoNetwork, TopoPopulation
import neuron  # needs to be imported before MPI
from time import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

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
# Initialization of MPI variables                   #
#################################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


# if True, execute full model. If False, do only the plotting.
# Network simulation results must exist.
PROPERRUN = True

##########################################################################
# Network parameters
##########################################################################

###############################################################################
# Instantiate a Mesocircuit object with parameters from the command line:
# the general data directory data_dir, the name of the experiment name_exp, and
# the ID of this parameterset ps_id.
# Previously evaluated parameters are loaded.

circuit = mesoframe.Mesocircuit(
    data_dir=sys.argv[-3], name_exp=sys.argv[-2], ps_id=sys.argv[-1],
    load_parameters=True)

sim_dict = circuit.sim_dict
net_dict = circuit.net_dict

path_parameters = os.path.join(circuit.data_dir_circuit, 'parameters')


# check if mod file for synapse model specified in expsyni.mod is loaded.
# if not, compile and load it.

nmodl_dir = os.path.join(os.path.dirname(mesocircuit.__file__), 'lfp')
try:
    assert neuron.load_mechanisms(nmodl_dir)
except AssertionError:
    if RANK == 0:
        cwd = os.getcwd()
        os.chdir(nmodl_dir)
        os.system('nrnivmodl')
        os.chdir(cwd)
    COMM.Barrier()
neuron.load_mechanisms(nmodl_dir)


##########################################################################
# set up the file destination
##########################################################################
# path_lfp_data = os.path.join(os.path.split(path_parameters)[0], 'lfp')
path_lfp_data = os.path.join(circuit.data_dir_circuit, 'lfp')
if RANK == 0:
    if not os.path.isdir(path_lfp_data):
        os.mkdir(path_lfp_data)

# wait for operation to finish
COMM.Barrier()

##########################################################################
# get ParameterSet object instance with all required parameters for LFPs etc.
##########################################################################
PS = get_parameters(path_lfp_data=path_lfp_data,
                    sim_dict=sim_dict,
                    net_dict=net_dict)

# create file for simulation time(s) to file
if RANK == 0:
    simstats = open(os.path.join(
        PS.savefolder, f'simstats_{sys.argv[1]}.dat'), 'w')
    simstats.write('task time\n')

# tic toc
tic = time()
ticc = tic


##########################################################################
# Create an object representation containing the spiking activity of the
# network simulation output that uses sqlite3.
##########################################################################
networkSim = CachedTopoNetwork(**PS.network_params)

# tic toc
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
    probes.append(VolumetricCurrentSourceDensity(cell=None, **PS.CSDParams))
    probes.append(CurrentDipoleMoment(cell=None))

##############################################################################
# Create multicompartment neuron populations for LFP predictions
##############################################################################
if PROPERRUN:
    # iterate over each cell type, and create populationulation object
    for i, y in enumerate(PS.y):
        if y == sys.argv[1]:
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
                simstats.write('Population {}\n'.format(tocc - ticc))

            # run population simulation and collect the data
            ticc = time()
            pop.run()
            tocc = time()
            if RANK == 0:
                simstats.write('run {}\n'.format(tocc - ticc))

            ticc = time()
            pop.collect_data()
            tocc = time()

            if RANK == 0:
                simstats.write('collect {}\n'.format(tocc - ticc))

            # object no longer needed
            del pop

# tic toc
print(('Execution time: %.3f seconds' % (time() - tic)))


COMM.Barrier()
