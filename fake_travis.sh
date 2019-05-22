
export SLAC_ST_BUILD=false
export PIP_DEPS='coverage pytest-cov'
export CONDA2='conda install -y -c conda-forge healpy'
export INSTALL_CMD='python setup.py install'
export CONDA_CHANNELS="conda-forge"
export CONDA_DEPS='gammapy numpy astropy scipy matplotlib pytest pyyaml'
export PYTHON_VERSION="2.7"
export CONDA_DOWNLOAD="Miniconda-latest-Linux-x86_64.sh"

export ST_INSTALL="conda install -y -c conda-forge/label/cf201901 -c fermi fermitools"
export CONDA2="conda install -y -c conda-forge healpy"
export INSTALL_CMD='python setup.py install'
export CONDA_PATH='/u/ek/echarles/dmcat/software/build_test/miniconda'

\rm -rf $CONDA_PATH
source condainstall.sh $CONDA_PATH
$ST_INSTALL

bash travistests.sh

coveralls --rcfile='fermipy/tests/coveragerc
