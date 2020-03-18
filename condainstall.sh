#!/usr/bin/env bash

if [[ -z $PYTHON_VERSION ]]; then
    PYTHON_VERSION=2.7
fi

if [[ -z $CONDA_DOWNLOAD ]]; then
    if [[ `uname` == "Darwin" ]]; then
	echo "Detected Mac OS X..."
	CONDA_DOWNLOAD=Miniconda3-latest-MacOSX-x86_64.sh
    else
	echo "Detected Linux..."
	CONDA_DOWNLOAD=Miniconda3-latest-Linux-x86_64.sh
    fi
fi

if [[ -z $CONDA2 ]]; then
    CONDA2='conda install -y healpy subprocess32'
fi

if [[ -z $CONDA_PATH ]]; then
    CONDA_PATH=$HOME/miniconda
fi

if [ ! -d "$CONDA_PATH/bin" ]; then
    echo Creating a new conda installation under $CONDA_PATH
    curl -o miniconda.sh -L http://repo.continuum.io/miniconda/$CONDA_DOWNLOAD
    bash miniconda.sh -b -p $CONDA_PATH
    rc=$?
    rm -f miniconda.sh
    if [[ $rc != 0 ]]; then
        exit $rc
    fi
    export PATH="$CONDA_PATH/bin:$PATH"
else
    echo "Using existing conda installation under $CONDA_PATH"
fi

if [[ -n $ST_INSTALL ]]; then
    $ST_INSTALL
fi

conda update -q conda -y
conda create --name fermipy-test-build -c conda-forge -y python=$PYTHON_VERSION
conda install -n fermipy-test-build -y pytest 
conda install -n fermipy-test-build --only_deps -y fermipy
conda activate fermipy-test-build

if [[ -n $CONDA2 ]]; then
    $CONDA2
fi

if [[ -n $PIP_DEPS ]]; then
    python -m pip install $PIP_DEPS
fi

source condasetup.sh

if [[ -n $INSTALL_CMD ]]; then
    $INSTALL_CMD
fi

