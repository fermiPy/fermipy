# Setup script for conda-based ST installations

unset LD_LIBRARY_PATH
unset DYLD_LIBRARY_PATH
unset PYTHONPATH

export PATH=$HOME/miniconda/bin:$PATH
source activate fermi-env
export PATH=$CONDA_ENV_PATH/bin:$PATH

if [[ $SLAC_ST_BUILD == true ]]; then
    source $FERMI_DIR/bin/redhat6-x86_64-64bit-gcc44-Optimized/_setup.sh
    export PYTHONPATH=$FERMI_DIR/python:$FERMI_DIR/lib/redhat6-x86_64-64bit-gcc44-Optimized
else
    source $FERMI_DIR/fermi-init.sh
    export PYTHONPATH=$FERMI_DIR/lib/python:$FERMI_DIR/lib
fi
    



