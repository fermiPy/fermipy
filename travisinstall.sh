curl -o miniconda.sh -L http://repo.continuum.io/miniconda/$CONDA_DOWNLOAD
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n fermi-env python=$PYTHON_VERSION pip numpy astropy pytest $CONDA_DEPS
source activate fermi-env
python -m pip install coverage pytest-cov coveralls
$CONDA2
python setup.py install
#$ST_INSTALL # Download and install the ST binaries
#source travissetup.sh
#source condasetup.sh

