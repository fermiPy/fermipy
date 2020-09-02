
export SLAC_ST_BUILD=false
export PIP_DEPS='coverage pytest-cov coveralls'
export INSTALL_CMD='python setup.py install'
export CONDA_CHANNELS="conda-forge"
export FERMI_CONDA_CHANNELS="-c conda-forge/label/cf201901 -c fermi"
export CONDA_DEPS='gammapy numpy astropy scipy matplotlib pyyaml astropy-healpix'
export CONDA2_DEPS='subprocess32 pytest'
export DOCKER_INSTALL=''

NAME='py37'
export FERMIPY_CONDA_ENV="fermipy-test-$NAME"


case $NAME in
    sandbox)
	export CONDA_PATH="/u/ek/echarles/dmcat/software/anaconda3"
        export FERMIPY_CONDA_ENV="fermipy-py36"
        export PYTHON_VERSION="3.6"
        export ST_INSTALL=" "
        export CONDA_DEPS='gammapy numpy astropy scipy matplotlib pytest pyyaml sphinx sphinx_rtd_theme'
	export PIP_DEPS=" "
	export CONDA2_DEPS=" "
        ;;
    main)
	export PYTHON_VERSION="2.7"
	export CONDA_PATH="/Users/echarles/miniconda2"
	export CONDA_DEPS='gammapy=0.10 healpy=1.13.0 numpy astropy scipy matplotlib pyyaml'	
	export ST_INSTALL="conda install -y --name $FERMIPY_CONDA_ENV -c conda-forge -c fermi -c fermi/label/beta fermitools"
	;;
    old)
	export PYTHON_VERSION="2.7"
	export CONDA_PATH="/Users/echarles/miniconda2"
	export CONDA_DEPS='gammapy=0.10 healpy=1.13.0 numpy astropy scipy matplotlib pyyaml'	
	export ST_INSTALL="conda install -y --name $FERMIPY_CONDA_ENV -c conda-forge/label/cf201901 -c fermi -c conda-forge fermitools=1.2.23"
	;;
    py37)
	export PYTHON_VERSION="3.7"
	export CONDA_PATH="/Users/echarles/miniconda3"
        export ST_INSTALL="conda install -y --name $FERMIPY_CONDA_ENV -c conda-forge -c fermi -c fermi/label/beta fermitools"
        export CONDA2_DEPS='pytest'
	;;
    docs)
	export PYTHON_VERSION="3.7"
	export CONDA_PATH="/Users/echarles/miniconda3"
        export ST_INSTALL=''
	export PIP_DEPS='coverage pytest-cov coveralls'
	export CONDA2_DEPS='pytest sphinx sphinx_rtd_theme'
	;;
    py37_st-no_dep)
	export PYTHON_VERSION="3.7"
	export CONDA_PATH="/Users/echarles/miniconda3"
        export ST_INSTALL=''
        export CONDA2_DEPS='pytest'	
	;;
    py2_st-no_dep)
	export PYTHON_VERSION="2.7"
	export CONDA_PATH="/Users/echarles/miniconda2"
	export CONDA_DEPS='gammapy=0.10 healpy=1.13.0 numpy astropy scipy matplotlib pyyaml'	
        export ST_INSTALL=''
	;;
    none)
	exit
        ;;
esac

echo Running fake_travis for build $NAME

source condainstall.sh 

bash travistests.sh

coveralls --rcfile='fermipy/tests/coveragerc'
