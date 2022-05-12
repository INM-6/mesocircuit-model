FROM buildpack-deps:focal

RUN apt-get update && \
    apt-get install -y \
    cmake \
    libmpich-dev \
    mpich \
    doxygen \
    libboost-dev \
    libgsl-dev \
    cython3 \
    python3-dev \
    python3-pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

RUN pip install mpi4py


# ----- Install NEST -----
RUN git clone https://github.com/nest/nest-simulator.git && \
    cd nest-simulator && \
    # git checkout master && \
    git checkout b628ccf && \
    cd .. && \
    mkdir nest-build && \
    ls -l && \
    cd  nest-build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/nest/ \
        -Dwith-ltdl=ON \
        -Dwith-gsl=ON \
        -Dwith-readline=ON \
        -Dwith-python=ON \
        -Dwith-mpi=ON \
        -Dwith-openmp=ON \
        -Dwith-boost=ON \
        ../nest-simulator && \
    make && \
    make install && \
    cd /

RUN rm -r nest-simulator
RUN rm -r nest-build


# ---- additional requirements
RUN apt-get install -y \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-pandas \
    ipython3 \
    jupyter

RUN update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10

# installing serial h5py (deb package installs OpenMPI which may conflict with MPICH)
RUN pip install h5py


# ---- install neuron -----
RUN pip install neuron


# ---- install LFPy@master -----
RUN pip install git+https://github.com/LFPy/LFPy.git@v2.2.6#egg=LFPy


# --- Install hybridLFPy@nest3 ----
RUN pip install git+https://github.com/INM-6/hybridLFPy.git@nest3#egg=hybridLFPy


# ---- Install additional dependencies ----
RUN pip3 install git+https://github.com/NeuralEnsemble/parameters@master

# Add NEST binary folder to PATH
ENV PATH /opt/nest/bin:${PATH}

# Add PyNEST to PYTHONPATH
ENV PYTHONPATH /opt/nest/lib/python3.8/site-packages:${PYTHONPATH}

# If running with Singularity, run the below line in the host.
# PYTHONPATH set here doesn't carry over:
# export SINGULARITYENV_PYTHONPATH=/opt/nest/lib/python3.8/site-packages
# Alternatively, run "source /opt/local/bin/nest_vars.sh" while running the container
