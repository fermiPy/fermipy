

# Set this to point at your conda installation
export CONDA_PATH=$HOME/miniconda2
# Set this to the name of the conda environment you want for fermipy development
export FERMIPY_CONDA_ENV=fermipy-dev

# Don't touch these
export INSTALL_CMD="python setup.py develop"
unset PYTHON_VERSION
unset CONDA2_DEPS
unset PIP_DEPS
unset ST_INSTALL

. condainstall.sh
