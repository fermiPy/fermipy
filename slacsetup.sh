# SLAC fermipy/ST environment setup script
#
# To use this script do:
# $ source slacsetup.sh
# $ slacsetup [STVERSION] [CONDABASE]
# 
# [STVERSION] : ST Release Tag or path to local ST installation.  If
# this argument is empty the ST version will be set with
# FERMI_RELEASE_TAG.
#
# [CONDABASE] : Override path to anaconda python installation.  Set
# this argument to use an anaconda installation different from the one
# in GLAST_EXT.
#

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

# Set default ST version
if [ -z $FERMI_RELEASE_TAG ]; then
    export FERMI_RELEASE_TAG=11-05-02
fi

# Save PATH
export PATH_BACKUP=$PATH

function slacsetup
{
    export GLAST_EXT=/afs/slac/g/glast/ground/GLAST_EXT/${BLDARCH}
    export BUILDS=/nfs/farm/g/glast/u35/ReleaseManagerBuild
    export PATH=/usr/lib64/qt-3.3/bin:/opt/lsf-openmpi/1.8.1/bin/:/usr/local/bin:/bin:/usr/bin:/usr/X11R6/bin

    unset PFILES 
    unset PYTHONPATH
    unset LD_LIBRARY_PATH
    unset DYLD_LIBRARY_PATH

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
    
    if [ -n "$2" ]; then
	export PATH=$2/bin:$PATH
    elif [ -n "$CONDABASE" ]; then
	export PATH=$CONDABASE/bin:$PATH
    else
	export PYTHONROOT=$GLAST_EXT/python/2.7.12-anaconda2-4.2.0
	export PYTHON_USER_BIN=$(python -c 'import site; print(site.USER_BASE + "/bin")')
	export PATH=$PYTHON_USER_BIN:$PYTHONROOT/bin:$PATH
    fi

    export FERMI_DIFFUSE_DIR=$GLAST_EXT/diffuseModels/v5r0

    # Setup HEASoft
    ASTROTOOLS=/afs/slac/g/glast/applications/astroTools
    export PATH=${PATH}:${ASTROTOOLS}/bin
    export LHEASOFT=${ASTROTOOLS}/headas/i686-pc-linux-gnu-libc2.2.4
    export HEADAS=${LHEASOFT}
    source ${HEADAS}/headas-init.sh
}
