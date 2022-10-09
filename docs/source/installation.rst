Installation
============

Create a `conda <https://conda.io>`_ environment:

.. code:: bash

    conda env create -f environment.yml
    conda activate mesocircuit


Installing
----------

The LFP predictions requires a module which should be installed as

::

   $ python setup.py develop --user

Docker
------

We provide a Docker (https://www.docker.com) container build file with
codes required for this project. To get started, install Docker and
issue

::

   # build local Dockerfile (obtained by cloning repo, checkout branch etc.)
   $ docker build -t <container> - < Dockerfile
   $ docker run -it -p 5000:5000 <container>:latest

The ``--mount`` option can be used to mount a folder on the host to a
target folder as:

::

   $ docker run --mount type=bind,source="$(pwd)",target=/opt/<target> -it -p 5000:5000 <container>:latest

which mounts the present working dirctory (``$(pwd)``) to the
``/opt/<target>`` directory of the container. Try mounting the
``mesocircuit`` source directory for example (by setting
``source="<path-to-mesocircuit>"``). Various files can then be found in
the folder ``/opt/mesocircuit`` when the container is running.

Python code can be run in parallel by issuing

::

   $ mpirun -n $(nproc) python <filename> <arguments>

Jupyter notebook servers running from within the container can be
accessed after invoking them by issuing:

::

   $ cd /opt/mesocircuit
   $ jupyter notebook --ip 0.0.0.0 --port=5000 --no-browser --allow-root

and opening the resulting URL in a browser on the host computer, similar
to:
http://127.0.0.1:5000/?token=dcf8f859f859740fc858c568bdd5b015e0cf15bfc2c5b0c1

.. _docker-1:

Docker
------

Docker (https://www.docker.com) provides a solution for packaging all
project requirements in a single container. This can be used for
simulations and analysis, and may be supported on the HPC resource
(e.g., via Singularity). Make sure the Docker client is running on the
host. Then:

::

   # build image using Docker:
   docker build -t mesocircuit - < Dockerfile

   # start container mounting local file system, then open a jupyter-notebook session:
   docker run --mount type=bind,source="$(pwd)",target=/opt/data -it -p 5000:5000 mesocircuit
   /# cd /opt/data/
   /# jupyter-notebook --ip 0.0.0.0 --port=5000 --no-browser --allow-root
   # take note of the URL printed to the terminal, and open it in a browser on the host.

   # oneliner (open URL, then browse to `/opt/data/` and open notebooks):
   docker run --mount type=bind,source="$(pwd)",target=/opt/data -it -p 5000:5000 mesocircuit jupyter-notebook --ip 0.0.0.0 --port=5000 --no-browser --allow-root
   # take note of the URL printed to the terminal, and open it in a browser on the host.

   # A working Python/MPI environment should be present in the running container. Hence scripts can be run interactively issuing:
   docker run --mount type=bind,source="$(pwd)",target=/opt/data -it -p 5000:5000 mesocircuit
   /# cd /opt/data/

   # start an interactive Python session
   /# ipython
   >>> import scipy  # etc
   >>> quit()

   # run a simulation with MPI, assuming we have access to 1024 physical CPU cores (also make sure that parameter files have been created by an earlier call to `python run_pscan.py`)
   /# mpiexec -n 1024 python task.py

Singularity
-----------

Singularity things (see
https://apps.fz-juelich.de/jsc/hps/jusuf/cluster/container-runtime.html):

Build singularity container ``lfpykernels.sif`` using the JSC build
system:

::

   module --force purge
   module load Stages/2022 GCCcore/.11.2.0 Apptainer-Tools/2022 GCC/11.2.0 ParaStationMPI/5.5.0-1
   sib upload ./Dockerfile mesocircuit
   sib build --recipe-name mesocircuit --blocking  # this will take a few minutes
   sib download --recipe-name mesocircuit

Compile NMODL files using the ``nrnivmodl`` script included in the
container:

::

   cd mod && rm -rf x86_64 && singularity exec ../mesocircuit.sif nrnivmodl && cd ..  # compile NMODL files for the container
   srun --mpi=pmi2 singularity exec mesocircuit.sif python3 -u task.py

Make sure that jobscripts are configured for singularity, calling the
built-in python executable: unset DISPLAY # matplotlib may look for a
nonexistant display on compute node(s) module –force purge module load
Stages/2022 GCCcore/.11.2.0 Apptainer-Tools/2022 GCC/11.2.0
ParaStationMPI/5.5.0-1 srun –mpi=pmi2 singularity exec mesocircuit.sif
python3 -u task.py # execute simulation
