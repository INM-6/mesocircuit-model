FROM buildpack-deps:jammy

# ---- install .deb packages ----
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    cmake=3.22.1-1ubuntu1.22.04.2 \
    libmpich-dev=4.0-3 \
    mpich=4.0-3 \
    doxygen=1.9.1-2ubuntu2 \
    libboost-dev=1.74.0.3ubuntu7 \
    libgsl-dev=2.7.1+dfsg-3 \
    libltdl-dev=2.4.6-15build2 \
    fonts-humor-sans=1.0-4 \
    cython3=0.29.28-1ubuntu3 \
    python3-dev=3.10.6-1~22.04 \
    python3-pip=22.0.2+dfsg-1 \
    python3-numpy=1:1.21.5-1ubuntu22.04.1 \ 
    python3-scipy=1.8.0-1exp2ubuntu1 \
    python3-pandas=1.3.5+dfsg-3 \
    python3-seaborn=0.11.2-3 \
    python3-astropy=5.0.2-1 \
    python3-yaml=5.4.1-1ubuntu1 \
    python3-sympy=1.9-1 \
    python3-matplotlib=3.5.1-2build1 \
    python3-autopep8=1.6.0-1 \
    python3-graphviz=0.14.2-1 \
    python3-prettytable=2.5.0-2 \
    ipython3=7.31.1-1 \
    antlr4=4.7.2-5 \
    libhdf5-dev=1.10.7+repack-4ubuntu2 \
    bison=2:3.8.2+dfsg-1build1 \
    flex=2.6.4-8build2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# fix some executable names
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10

# ---- install mpi4py and h5py (deb packages may depend on libopenmpi which we do not want) ----
RUN pip --no-cache-dir install --no-deps mpi4py==3.1.3 && \
  pip --no-cache-dir install --no-deps h5py==3.7.0 && \
  pip --no-cache-dir install jupyterlab==3.4.3

# ---- install NEST 3.4 ----
RUN git clone --depth 1 -b v3.8 https://github.com/nest/nest-simulator /usr/src/nest-simulator && \
  mkdir nest-build && \
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/ \
        -Dwith-optimize="-O2" \
        -Dwith-warning=ON \
        -Dwith-boost=ON \
        -Dwith-ltdl=ON \
        -Dwith-gsl=ON \
        -Dwith-readline=ON \
        -Dwith-python=ON \
        -Dwith-mpi=ON \
        -Dwith-openmp=ON \
        -Dwith-libneurosim=OFF \
        -Dwith-sionlib=OFF \
        -Dwith-music=OFF \
        -Dwith-hdf5=OFF \
        -S /usr/src/nest-simulator \
        -B nest-build && \
  make -C nest-build && \
  make -C nest-build install && \
  # clean up install/build files
  rm -r /usr/src/nest-simulator && \
  rm -r nest-build && \
  # Add NEST binary folder to PATH
  echo "source /usr/local/bin/nest_vars.sh" >> root/.bashrc

# ---- install NEURON ----
RUN git clone --depth 1 -b 8.2.2 https://github.com/neuronsimulator/nrn.git /usr/src/nrn && \
  mkdir nrn-bld && \
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/ \
    -DCURSES_NEED_NCURSES=ON \
    -DNRN_ENABLE_INTERVIEWS=OFF \
    -DNRN_ENABLE_MPI=ON \
    -DNRN_ENABLE_RX3D=OFF \
    -DNRN_ENABLE_PYTHON=ON \
    -DNRN_MODULE_INSTALL_OPTIONS="" \
    -S /usr/src/nrn \
    -B nrn-bld && \
  cmake --build nrn-bld --parallel 4 --target install && \
  # clean up
  rm -r /usr/src/nrn && \
  rm -r nrn-bld

# ---- install LFPy, hybridLFPy etc. -----
RUN pip --no-cache-dir install --no-deps git+https://github.com//alejoe91/MEAutility.git@1.5.1#egg=MEAutility && \
  pip --no-cache-dir install --no-deps git+https://github.com/LFPy/LFPykit.git@v0.5#egg=lfpykit && \
  pip --no-cache-dir install --no-deps git+https://github.com/LFPy/LFPy.git@5d241d62080f881415fd4becae06c8107571d2d1 && \
  pip --no-cache-dir install --no-deps git+https://github.com/INM-6/hybridLFPy.git@4254e56d581e9b1f48f696853c3d969c4e561d8b && \
  pip --no-cache-dir install --no-deps git+https://github.com/NeuralEnsemble/parameters@b95bac2bd17f03ce600541e435e270a1e1c5a478 && \
  pip --no-cache-dir install --no-deps git+https://github.com/INM-6/nnmt.git@v1.3.0#egg=nnmt

RUN pip --no-cache-dir install --no-deps pytest

# ---- install meso_analysis package ----
RUN pip --no-cache-dir install --no-deps git+https://github.com/INM-6/mesocircuit-model.git#egg=mesocircuit

# If running with Singularity/Apptainer, run the below line in the host.
# PYTHONPATH set here doesn't carry over:
# export SINGULARITYENV_PYTHONPATH=/opt/nest/lib/python3.8/site-packages
# Alternatively, run "source /opt/local/bin/nest_vars.sh" while running the container
