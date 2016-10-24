FROM mdwood/fermist:11-04-00
MAINTAINER Matthew Wood <mdwood@slac.stanford.edu>
ARG CONDA_DOWNLOAD=Miniconda-latest-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    curl -o miniconda.sh -L http://repo.continuum.io/miniconda/$CONDA_DOWNLOAD && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && conda update -y conda && conda config --set channel_priority false 
RUN conda install -y pip numpy astropy matplotlib scipy pytest ipython jupyter && \
    conda install -y -c conda-forge wcsaxes healpy fermipy
