
export SLAC_ST_BUILD=false
export PIP_DEPS='coverage pytest-cov coveralls'
export INSTALL_CMD='python setup.py install'
export CONDA_CHANNELS="conda-forge"
export FERMI_CONDA_CHANNELS="-c conda-forge/label/cf201901 -c fermi"
export CONDA_DEPS='gammapy numpy astropy scipy matplotlib pyyaml astropy-healpix'
export CONDA2_DEPS='subprocess32 pytest'
export DOCKER_INSTALL=''

NAME='docs'
export FERMIPY_CONDA_ENV="fermipy-test-$NAME"


case $NAME in
    main)
	export PYTHON_VERSION="2.7"
	export CONDA_PATH="/Users/echarles/anaconda2"
	export ST_INSTALL="conda install -y --name $FERMIPY_CONDA_ENV $FERMI_CONDA_CHANNELS -c $CONDA_CHANNELS -c fermi fermitools"
	;;
    old)
	export PYTHON_VERSION="2.7"
	export CONDA_PATH="/Users/echarles/anaconda2"
	export ST_INSTALL="conda install -y --name $FERMIPY_CONDA_ENV $FERMI_CONDA_CHANNELS -c $CONDA_CHANNELS -c fermi fermitools=1.2.23"
	;;
    docs)
	export PYTHON_VERSION="3.6"
	export CONDA_PATH="/Users/echarles/anaconda3"
        export ST_INSTALL=''
        export CONDA_DEPS='gammapy numpy astropy scipy matplotlib pytest pyyaml sphinx sphinx_rtd_theme'
	;;
    py36_st-no_dep)
	export PYTHON_VERSION="3.6"
	export CONDA_PATH="/Users/echarles/anaconda3"
        export ST_INSTALL=''
	;;
    py2_st-no_dep)
	export PYTHON_VERSION="2.7"
	export CONDA_PATH="/Users/echarles/anaconda2"
        export ST_INSTALL=''
	;;
    slac*)
	export PYTHON_VERSION="2.7"
	export ST_INSTALL=""
	export SLAC_ST_BUILD=true
	export INST_DIR='/u/ek/echarles/dmcat/software/git-releases/FT_01-00-01_orig'
	;;
    none)
	exit
        ;;
esac

echo Running fake_travis for build $NAME

source condainstall.sh 

bash travistests.sh

coveralls --rcfile='fermipy/tests/coveragerc'
