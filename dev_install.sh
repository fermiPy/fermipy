

# Set this to point at your conda installation
export CONDA_PATH=$HOME/anaconda2
# Set this to the name of the conda environment you want for fermipy development
export FERMI_CONDA_ENV=fermipy-dev-test

# Don't touch these
export INSTALL_CMD="python setup.py develop"
unset PYTHON_VERSION
unset CONDA2_DEPS
unset PIP_DEPS
unset ST_INSTALL

. condainstall.sh
