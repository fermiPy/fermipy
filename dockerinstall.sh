docker build -t mdwood/fermist .
docker info
docker images
docker run -it -d -v $PWD:/home/fermipy --name=test0 \
       -e GLAST_EXT=/home/externals \
       -e INST_DIR=/home \
       -e CONDA_DOWNLOAD="$CONDA_DOWNLOAD" \
       -e CONDA_DEPS="$CONDA_DEPS" \
       -e PIP_DEPS="$PIP_DEPS" \
       -e PYTHON_VERSION="$PYTHON_VERSION" \
       -e CONDA2="$CONDA2" \
       -e SLAC_ST_BUILD="$SLAC_ST_BUILD" \
       mdwood/fermist /bin/bash

