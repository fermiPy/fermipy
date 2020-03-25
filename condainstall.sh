#!/usr/bin/env bash

# Default python version
if [[ -z $PYTHON_VERSION ]]; then
    PYTHON_VERSION=2.7
fi

# Default conda download
if [[ -z $CONDA_DOWNLOAD ]]; then
    if [[ `uname` == "Darwin" ]]; then
	echo "Detected Mac OS X..."
	CONDA_DOWNLOAD=Miniconda3-latest-MacOSX-x86_64.sh
    else
	echo "Detected Linux..."
	CONDA_DOWNLOAD=Miniconda3-latest-Linux-x86_64.sh
    fi
fi

# Default conda deps
if [[ -z $CONDA_DEPS ]]; then
    CONDA_DEPS='scipy matplotlib pyyaml numpy astropyy gammapy healpy'
fi

# Default conda path
if [[ -z $CONDA_PATH ]]; then
    CONDA_PATH=$HOME/miniconda
fi


# Default conda env
if [[ -z $FERMIPY_CONDA_ENV ]]; then
    FERMIPY_CONDA_ENV=fermipy
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

# Make sure we have conda setup
. $CONDA_PATH/etc/profile.d/conda.sh

# First we update conda and make an env
conda update -q conda -y
conda create --name $FERMIPY_CONDA_ENV $FERMI_CONDA_CHANNELS -y python=$PYTHON_VERSION
conda activate $FERMIPY_CONDA_ENV

# Install the science tools, if requested
if [[ -n $ST_INSTALL ]]; then
    $ST_INSTALL
fi

# Ok, now we install fermipy dependencies
conda install -n $FERMIPY_CONDA_ENV -y -c conda-forge $CONDA_DEPS

# Install extra conda deps if needed
if [[ -n $CONDA2_DEPS ]]; then
    conda install -n $FERMIPY_CONDA_ENV -y -c conda-forge $CONDA2_DEPS
fi

# Install stuff from pip (for travis coverage testing)
if [[ -n $PIP_DEPS ]]; then
    python -m pip install $PIP_DEPS
fi

source condasetup.sh

# We don't set this up by default.  Let the user specify how to do it.
# This is just here from the travis testing
if [[ -n $INSTALL_CMD ]]; then
    $INSTALL_CMD
fi

