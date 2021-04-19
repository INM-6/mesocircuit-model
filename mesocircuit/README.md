Mesocircuit
===========

Installing
----------

The LFP predictions requires a module which should be installed as

    $ python setup.py develop --user


Docker
------

We provide a Docker (https://www.docker.com) container build file with codes required for this project.
To get started, install Docker and issue

    # build local Dockerfile (obtained by cloning repo, checkout branch etc.)
    $ docker build -t <container> - < Dockerfile
    $ docker run -it -p 5000:5000 <container>:latest


The ``--mount`` option can be used to mount a folder on the host to a target folder as:

    $ docker run --mount type=bind,source="$(pwd)",target=/opt/<target> -it -p 5000:5000 <container>:latest

which mounts the present working dirctory (``$(pwd)``) to the ``/opt/<target>`` directory of the container.
Try mounting the ``mesocircuit`` source directory for example (by setting ``source="<path-to-mesocircuit>"``).
Various files can then be found in the folder ``/opt/mesocircuit``
when the container is running.

Python code can be run in parallel by issuing

    $ mpirun -n $(nproc) python <filename> <arguments>

Jupyter notebook servers running from within the
container can be accessed after invoking them by issuing:

    $ cd /opt/mesocircuit
    $ jupyter notebook --ip 0.0.0.0 --port=5000 --no-browser --allow-root

and opening the resulting URL in a browser on the host computer, similar to:
http://127.0.0.1:5000/?token=dcf8f859f859740fc858c568bdd5b015e0cf15bfc2c5b0c1


Singularity
-----------

Confer JSC instructions on how to build container at https://gitlab.version.fz-juelich.de/bvonstvieth_publications/container_userdoc_tmp
