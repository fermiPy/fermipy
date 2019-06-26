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

if [[ -z $CONDA_DEPS ]]; then
    CONDA_DEPS='scipy matplotlib pyyaml ipython numpy astropy'
fi

if [[ -z $CONDA2 ]]; then
    CONDA2='conda install -y healpy subprocess32 fermipy jupyter'
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
else
    echo "Using existing conda installation under $CONDA_PATH"
fi

export PATH="$CONDA_PATH/bin:$PATH"

conda update -q conda -y
#conda config --add channels conda-forge
conda info -a
conda install -c conda-forge -y python=$PYTHON_VERSION pip pytest $CONDA_DEPS

if [[ -n $CONDA2 ]]; then
    $CONDA2
fi

if [[ -n $PIP_DEPS ]]; then
    python -m pip install $PIP_DEPS
fi

if [[ -n $INSTALL_CMD ]]; then
    $INSTALL_CMD
fi

