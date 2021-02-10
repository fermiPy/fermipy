
export SLAC_ST_BUILD=false
export PIP_DEPS='coverage pytest-cov'
export PYTHON_VERSION=3.7
#export INSTALL_CMD='python setup.py install'
export INSTALL_CMD='pip install . --no-deps'
export CONDA_PATH="$HOME/miniconda3"
export CONDA_CHANNELS="conda-forge"
export CONDA_DOWNLOAD="Miniconda3-latest-Linux-x86_64.sh"
export CONDA_DEPS='gammapy numpy astropy<4 scipy matplotlib pyyaml astropy-healpix healpy'
export CONDA2_DEPS='subprocess32 pytest curl'
#export FERMI_CONDA_CHANNELS="-c conda-forge/label/cf201901 -c fermi -c fermi/label/beta"
export FERMI_CONDA_CHANNELS="-c conda-forge -c fermi"
export DOCKER_INSTALL=''

NAME='main'
export FERMIPY_CONDA_ENV="fermipy-test-$NAME"


case $NAME in
    main)
	export ST_INSTALL="conda install -y --name $FERMIPY_CONDA_ENV -c conda-forge -c fermi fermitools>=2"
	;;
    dev)
	export ST_INSTALL="conda install -y --name $FERMIPY_CONDA_ENV -c conda-forge -c fermi -c fermi/label/dev fermitools>=2"
	;;
    no-deps)
        export ST_INSTALL="echo"
        export CONDA2_DEPS='pytest curl'
	;;
    docs)
        export ST_INSTALL='echo'
	export PIP_DEPS='coverage pytest-cov coveralls'
	export CONDA2_DEPS='pytest sphinx sphinx_rtd_theme curl'
	;;
    none)
	exit
        ;;
esac

echo Running fake_travis for build $NAME

source condainstall.sh 

bash travistests.sh

#coveralls --rcfile='fermipy/tests/coveragerc'
