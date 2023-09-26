Installation
============

High-performance computing is needed for running the full mesocircuit model.
To set up the compute environment, refer to :ref:`hpc-label`.

For testing locally (e.g., on a laptop), we recommend to use the provided conda environment following the steps described in :ref:`local-label`.
Note that only downscaled versions of the model can be simulated locally.

As an alternative, the mesocircuit framework can also be obtained va docker.

.. _hpc-label:

Setup on HPC system
-------------------

These instructions apply to the system JURECA-DC but can be adjusted for any
other compute cluster.

Load modules and general exports:

::

   module load Stages/2023 StdEnv/2023 CMake GCC GSL jemalloc Boost ParaStationMPI Python SciPy-Stack mpi4py h5py

   jutil env activate -p <XXX>
   export USERNAME=<XXX>
   export PROJ=$PROJECT/$USERNAME
   export SCRA=$SCRATCH/$USERNAME

   export REPO_DIR=$PROJ/repositories
   export SOFT_DIR=$PROJ/software
   export PY_DIR=$SOFT_DIR/pip_install

Install further Python packages via pip:

::

   pip install --prefix=$PY_DIR nnmt parameters
   pip install --prefix=$PY_DIR git+https://github.com/LFPy/LFPy@v2.3
   pip install --prefix=$PY_DIR git+https://github.com/INM-6/hybridLFPy@master

Then install and source :ref:`nest-label`.

Finally, navigate to the repository `mesocircuit-model` and run:

::

   pip install -e .

.. _local-label:

Local setup
-----------

Create a `conda <https://conda.io>`_ environment:

.. code:: bash

    conda env create -f environment.yml
    conda activate mesocircuit

The conda environment does not contain NEST because a NEST version configured with MPI is needed.
Therefore, install NEST from source when the conda environment is activated.
Define folders, e.g.,

::

   export REPO_DIR=$HOME/repositories
   export SOFT_DIR=$HOME/software 

and follow this description :ref:`nest-label`.


.. _nest-label:

NEST
----

::

   cd $REPO_DIR
   git clone https://github.com/nest/nest-simulator.git
   git checkout tags/v3.4

   export NEST_SRC_DIR=$REPO_DIR/nest-simulator
   export NEST_BUILD_DIR=$SOFT_DIR/nest/nest_3_4/build
   export NEST_INSTALL_DIR=$SOFT_DIR/nest/nest_3_4/install

   mkdir -p $NEST_BUILD_DIR
   cd $NEST_BUILD_DIR

   cmake -DCMAKE_INSTALL_PREFIX:PATH=$NEST_INSTALL_DIR -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++ -Dwith-mpi=ON -Dwith-boost=ON $NEST_SRC_DIR

   make -j && make install

   source $NEST_INSTALL_DIR/bin/nest_vars.sh


