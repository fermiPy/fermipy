# Setup script for conda-based ST installations

unset LD_LIBRARY_PATH
unset DYLD_LIBRARY_PATH
unset PYTHONPATH

if [[ $SLAC_ST_BUILD == true ]] || [[ ! -e $FERMI_DIR/fermi-init.sh ]]; then
    source $INST_DIR/bin/redhat6-x86_64-64bit-gcc44-Optimized/_setup.sh
    export PYTHONPATH=$INST_DIR/python:$INST_DIR/lib/redhat6-x86_64-64bit-gcc44-Optimized
else
    source $FERMI_DIR/fermi-init.sh
    export PYTHONPATH=$FERMI_DIR/lib/python:$FERMI_DIR/lib
fi
    
export PATH=$HOME/miniconda/bin:$PATH

if [[ -n $CONDA_ENV_PATH ]]; then
   export PATH=$CONDA_ENV_PATH/bin:$PATH
fi

