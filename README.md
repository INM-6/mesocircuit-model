# Mesocircuit Model

As long as this repository is private, build the documentation by running:

```
conda env create -f environment.yml
cd docs
make html
```


## Installation

### Docker

For reference we here provide a ``Dockerfile`` for building a [Docker](https://docker.com) image with all project dependencies.
To build the image run:

```bash
docker build --platform=linux/amd64 -t mesocircuit -f Dockerfile .
```

To run the image in a container run:

```bash
docker run --platform=linux/amd64 -it mesocircuit <binary>
```


```bash
The container recipe should also work with build systems for other containerization technologies such as [Singularity](https://sylabs.io/singularity/), see also [https://apps.fz-juelich.de/jsc/hps/jureca/container-runtime.html](https://apps.fz-juelich.de/jsc/hps/jureca/container-runtime.html).
