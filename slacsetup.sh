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
export FERMI_RELEASE_TAG=10-01-01

function slacsetup
{
    # Setup CVS
    export CVSROOT=/nfs/slac/g/glast/ground/cvs    
    export GLAST_EXT=/afs/slac/g/glast/ground/GLAST_EXT/${BLDARCH}
    export BUILDS=/nfs/farm/g/glast/u35/ReleaseManagerBuild
    export LD_LIBRARY_PATH=$GLAST_EXT/cfitsio/v3290-gl1/lib

    unset PFILES 
    unset PYTHONPATH

    if [ -e "${BUILDS}/${BLDARCH}/Optimized/ScienceTools/$1/bin/${BLDARCH}-Optimized/_setup.sh" ]; then
	export INST_DIR=${BUILDS}/${BLDARCH}/Optimized/ScienceTools/$1
	source ${INST_DIR}/bin/${BLDARCH}-Optimized/_setup.sh
    elif [ -n "$1" ]; then
	export INST_DIR=${BUILDS}/${BLDARCH}/Optimized/ScienceTools/$1
	source ${INST_DIR}/bin/${BLDARCH}-Optimized/_setup.sh
    else
	export INST_DIR=${BUILDS}/${BLDARCH}/Optimized/ScienceTools/${FERMI_RELEASE_TAG}
	source ${INST_DIR}/bin/${BLDARCH}-Optimized/_setup.sh
    fi
    
    export PYTHONROOT=$GLAST_EXT/python/2.7.10

    export PATH=$PYTHONROOT/bin:$PATH
    export PYTHONPATH=$PYTHONROOT/lib/python2.7/site-packages:$PYTHONPATH
    export LD_LIBRARY_PATH=$PYTHONROOT/lib/python2.7/site-packages:$LD_LIBRARY_PATH
    export GLASTSETUP=1
}
