# Setup script for conda-based ST installations

unset LD_LIBRARY_PATH
unset DYLD_LIBRARY_PATH
unset PYTHONPATH

if [[ -z $CONDA_PATH ]]; then
    CONDA_PATH=$HOME/miniconda
fi

if [[ $SLAC_ST_BUILD == true ]]; then
    source /afs/slac.stanford.edu/g/glast/ground/scripts/group_scons.sh
    source $INST_DIR/bin/redhat6-x86_64-64bit-gcc44-Optimized/_setup.sh
    export PYTHONPATH=$INST_DIR/python:$INST_DIR/lib/redhat6-x86_64-64bit-gcc44-Optimized
elif [[ -e $FERMI_DIR/fermi-init.sh ]]; then    
    source $FERMI_DIR/fermi-init.sh
    export PYTHONPATH=$FERMI_DIR/lib/python:$FERMI_DIR/lib
fi

