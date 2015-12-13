
if [ ! -d "$HOME/miniconda" ]; then
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
fi

export PATH="$HOME/miniconda/bin:$PATH"
conda create -q -n fermi-env -y python=2.7 numpy scipy matplotlib astropy
source activate fermi-env
pip install healpy
#python setup.py install
