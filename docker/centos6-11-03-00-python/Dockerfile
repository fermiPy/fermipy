FROM mdwood/fermist:11-03-00
MAINTAINER Matthew Wood <mdwood@slac.stanford.edu>
ARG PYTHON_VERSION=2.7
ARG CONDA_DOWNLOAD=Miniconda-latest-Linux-x86_64.sh
ARG CONDA_DEPS="scipy matplotlib pyyaml ipython jupyter"
ARG CONDA_DEPS_EXTRA="wcsaxes healpy fermipy"
ARG PIP_DEPS=""
ENV PATH /opt/conda/bin:$PATH
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    curl -o miniconda.sh -L http://repo.continuum.io/miniconda/$CONDA_DOWNLOAD && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && conda update -y conda && conda config --set channel_priority false 
RUN conda create -q -n fermi-env -y python=$PYTHON_VERSION pip numpy astropy pytest $CONDA_DEPS && \
    source activate fermi-env && \
    conda install -y -c conda-forge $CONDA_DEPS_EXTRA && \
    if [[ -n $PIP_DEPS ]]; then pip install $PIP_DEPS; fi
RUN mkdir /opt/notebooks
ENV SLAC_ST_BUILD=true
ENV INST_DIR=/home/ScienceTools
ENV GLAST_EXT=/home/externals
RUN echo "source $INST_DIR/bin/redhat6-x86_64-64bit-gcc44-Optimized/_setup.sh" >> /root/.bashrc && \
    echo "source activate fermi-env" >> /root/.bashrc
CMD ["/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser"]
ENTRYPOINT ["/bin/bash","--login","-c"]
