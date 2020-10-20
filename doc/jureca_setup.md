# Setup on JURECA

## Load modules and general exports

```
module use $OTHERSTAGES
module load Stages/2019a
module load GCC CMake ParaStationMPI GSL jemalloc h5py mpi4py

jutil env activate -p <XXX>
export USERNAME=<XXX>
export PROJ=$PROJECT/$USERNAME
export SCRA=$SCRATCH/$USERNAME

export REPO_DIR=$PROJ/repositories
export SOFT_DIR=$PROJ/software
export PY_DIR=$SOFT_DIR/custom_python

```

## Install NEST

```
mkdir -p $REPO_DIR
cd $REPO_DIR
git clone https://github.com/nest/nest-simulator.git
git checkout b628ccf
export NEST_SRC_DIR=$REPO_DIR/nest-simulator

cd NEST_SRC_DIR
export COMMIT=$(git rev-parse --short=7 HEAD)
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export NAME=$BRANCH-$COMMIT
echo $NAME

export NEST_BUILD_DIR=$SOFT_DIR/nest/$NAME/build
export NEST_INSTALL_DIR=$SOFT_DIR/nest/$NAME/install

mkdir -p $NEST_BUILD_DIR
cd $NEST_BUILD_DIR

cmake -DCMAKE_INSTALL_PREFIX:PATH=$NEST_INSTALL_DIR -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++ -Dwith-mpi=ON -Dwith-boost=ON $NEST_SRC_DIR

make -j && make install

source $NEST_INSTALL_DIR/bin/nest_vars.sh
```

## Custom Python packages

```
pip install --install-option="--prefix=$PY_DIR" parameters
```
