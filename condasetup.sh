# Setup script for conda-based ST installations

unset LD_LIBRARY_PATH
unset DYLD_LIBRARY_PATH
unset PYTHONPATH

if [[ $SLAC_ST_BUILD == true ]]; then
    source $INST_DIR/bin/redhat6-x86_64-64bit-gcc44-Optimized/_setup.sh
    export PYTHONPATH=$INST_DIR/python:$INST_DIR/lib/redhat6-x86_64-64bit-gcc44-Optimized
elif [[ -e $FERMI_DIR/fermi-init.sh ]]; then
    source $FERMI_DIR/fermi-init.sh
    export PYTHONPATH=$FERMI_DIR/lib/python:$FERMI_DIR/lib
else    
    export CONDA_PREFIX=$CONDA_PATH
    source $CONDA_PREFIX/etc/conda/activate.d/activate_fermitools.sh
fi

export PATH=$CONDA_PATH/bin:$PATH

if [[ -n $CONDA_ENV_PATH ]]; then
   export PATH=$CONDA_ENV_PATH/bin:$PATH
fi

