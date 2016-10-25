#!/usr/bin/env bash

if [[ -z $PYTHON_VERSION ]]; then
    PYTHON_VERSION=2.7
fi

if [[ -z $CONDA_DOWNLOAD ]]; then
    CONDA_DOWNLOAD=Miniconda-latest-Linux-x86_64.sh
fi

if [[ -z $CONDA_DEPS ]]; then
    CONDA_DEPS='scipy matplotlib pyyaml ipython'
fi

if [[ -z $CONDA2 ]]; then
    CONDA2='conda install -c conda-forge -y wcsaxes healpy'
fi

# Check if conda exists if not then install it
if ! type "conda" &> /dev/null; then

    if [ -n "$1" ]; then
	CONDA_PATH=$1
    elif [[ -z $CONDA_PATH ]]; then
	CONDA_PATH=$HOME/miniconda
    fi
    
    if [ ! -d "$CONDA_PATH" ]; then
	echo "Creating a new conda installation under $CONDA_PATH"
	curl -o miniconda.sh -L http://repo.continuum.io/miniconda/$CONDA_DOWNLOAD
	bash miniconda.sh -b -p $CONDA_PATH
    fi

    export PATH="$CONDA_PATH/bin:$PATH"
fi

conda update -q conda -y
conda config --set channel_priority false 
conda info -a
conda create -q -n fermi-env -y python=$PYTHON_VERSION pip numpy astropy pytest $CONDA_DEPS
source activate fermi-env

if [[ -n $CONDA2 ]]; then
    $CONDA2
fi

if [[ -n $PIP_DEPS ]]; then
    python -m pip install $PIP_DEPS
fi

if [[ -n $INSTALL_CMD ]]; then
    $INSTALL_CMD
fi

#pip install fermipy
