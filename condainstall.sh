#!/usr/bin/env bash

# Check if conda exists if not then install it
if ! type "conda" &> /dev/null; then

    if [ -n "$1" ]; then
	CONDA_PATH=$1
    else
	CONDA_PATH="$HOME/miniconda"
    fi
    
    if [ ! -d "$CONDA_PATH" ]; then
	echo 'Creating a new conda installation under $CONDA_PATH' 
	curl -o miniconda.sh -L http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
	bash miniconda.sh -b -p $CONDA_PATH
    fi

    export PATH="$CONDA_PATH/bin:$PATH"
fi

conda create -q -n fermi-env -y python=2.7 numpy scipy matplotlib astropy pytest pyyaml ipython
source activate fermi-env
conda install -c openastronomy -y wcsaxes
pip install healpy
pip install fermipy
