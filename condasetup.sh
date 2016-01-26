unset LD_LIBRARY_PATH
unset PYTHONPATH

source $FERMI_DIR/fermi-init.sh
export PATH=$HOME/miniconda/bin:$PATH
source activate fermi-env
export PATH=$CONDA_ENV_PATH/bin:$PATH
export PYTHONPATH=$FERMI_DIR/lib/python:$FERMI_DIR/lib
