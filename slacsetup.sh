# BLDARCH is defined for convenience
if [ $(fs sysname | grep -c rhel60) -gt 0 ]; then
    export BLDARCH=redhat6-x86_64-64bit-gcc44
elif [ $(fs sysname | grep -c amd64) -gt 0 ]; then
# 64-bit redhat5 at SLAC.
    export BLDARCH=redhat5-x86_64-64bit-gcc41
else
# 32-bit redhat5 at SLAC.
    export BLDARCH=redhat5-i686-32bit-gcc41
fi

# default ST version
export FERMI_RELEASE_TAG=11-03-00

function slacsetup
{
    export GLAST_EXT=/afs/slac/g/glast/ground/GLAST_EXT/${BLDARCH}
    export BUILDS=/nfs/farm/g/glast/u35/ReleaseManagerBuild
    export PATH=/usr/lib64/qt-3.3/bin:/opt/lsf-openmpi/1.8.1/bin/:/usr/local/bin:/bin:/usr/bin:/usr/X11R6/bin
    #export LD_LIBRARY_PATH=$GLAST_EXT/cfitsio/v3290-gl1/lib

    unset PFILES 
    unset PYTHONPATH
    unset LD_LIBRARY_PATH

    if [ -e "${BUILDS}/${BLDARCH}/Optimized/ScienceTools/$1/bin/${BLDARCH}-Optimized/_setup.sh" ]; then
	export INST_DIR=${BUILDS}/${BLDARCH}/Optimized/ScienceTools/$1
	source ${INST_DIR}/bin/${BLDARCH}-Optimized/_setup.sh
    elif [ -e "$1/bin/${BLDARCH}-Optimized/_setup.sh" ]; then
	export INST_DIR=$1
	source ${INST_DIR}/bin/${BLDARCH}-Optimized/_setup.sh
    elif [ -e "$1/bin/${BLDARCH}-Debug-Optimized/_setup.sh" ]; then
	export INST_DIR=$1
	source ${INST_DIR}/bin/${BLDARCH}-Debug-Optimized/_setup.sh
    elif [ -e "$1/bin/${BLDARCH}-Debug/_setup.sh" ]; then
	export INST_DIR=$1
	source ${INST_DIR}/bin/${BLDARCH}-Debug/_setup.sh
    elif [ -n "$1" ]; then
	export INST_DIR=${BUILDS}/${BLDARCH}/Optimized/ScienceTools/$1
	source ${INST_DIR}/bin/${BLDARCH}-Optimized/_setup.sh
    else
	export INST_DIR=${BUILDS}/${BLDARCH}/Optimized/ScienceTools/${FERMI_RELEASE_TAG}
	source ${INST_DIR}/bin/${BLDARCH}-Optimized/_setup.sh
    fi
    
    export PYTHONROOT=$GLAST_EXT/python/2.7.12
    export PYTHON_USER_BIN=$(python -c 'import site; print(site.USER_BASE + "/bin")')
    export PATH=$PYTHON_USER_BIN:$PYTHONROOT/bin:$PATH
    export GLASTSETUP=1
}
