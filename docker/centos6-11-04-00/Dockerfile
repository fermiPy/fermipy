FROM centos:6
MAINTAINER Matthew Wood <mdwood@slac.stanford.edu>
RUN yum install -y git libgomp libXext libSM libXrender libicu xorg-x11-server-Xvfb
ARG SLAC_EXTERNALS_URL="http://www.slac.stanford.edu/exp/glast/ground/software/nfsLinks/u35/externals/redhat6-x86_64-64bit-gcc44"
ARG SLAC_ST_URL="http://www.slac.stanford.edu/exp/glast/ground/software/nfsLinks/u35/ReleaseManagerBuild/redhat6-x86_64-64bit-gcc44/Optimized/ScienceTools"
RUN mkdir -p /home/externals \
    && curl -o /home/externals/CLHEP-2.1.0.1-gl1.tar.gz ${SLAC_EXTERNALS_URL}/CLHEP-2.1.0.1-gl1.tar.gz \
    && curl -o /home/externals/fftw-3.1.2-gl1.tar.gz ${SLAC_EXTERNALS_URL}/fftw-3.1.2-gl1.tar.gz \
    && curl -o /home/externals/gsl-1.16.tar.gz ${SLAC_EXTERNALS_URL}/gsl-1.16.tar.gz \
    && curl -o /home/externals/healpix-3.30.tar.gz ${SLAC_EXTERNALS_URL}/healpix-3.30.tar.gz \
    && curl -o /home/externals/wcslib-4.25.1-gl1.tar.gz ${SLAC_EXTERNALS_URL}/wcslib-4.25.1-gl1.tar.gz \
    && curl -o /home/externals/xerces-3.1.3.tar.gz ${SLAC_EXTERNALS_URL}/xerces-3.1.3.tar.gz \
    && curl -o /home/externals/cfitsio-v3370.tar.gz ${SLAC_EXTERNALS_URL}/cfitsio-v3370.tar.gz \
    && curl -o /home/externals/ape-2.8.tar.gz ${SLAC_EXTERNALS_URL}/ape-2.8.tar.gz \
    && curl -o /home/externals/f2c-3.4-gl4.tar.gz ${SLAC_EXTERNALS_URL}/f2c-3.4-gl4.tar.gz \
    && curl -o /home/externals/cppunit-1.10.2-gl1.tar.gz ${SLAC_EXTERNALS_URL}/cppunit-1.10.2-gl1.tar.gz \
    && curl -o /home/externals/ROOT-v5.34.34.tar.gz ${SLAC_EXTERNALS_URL}/ROOT-v5.34.34.tar.gz \
    && curl -o /home/externals/diffuseModels-v5r0.tar.gz ${SLAC_EXTERNALS_URL}/diffuseModels-v5r0.tar.gz \
    && cd /home/externals && for f in *.tar.gz; do tar -xf "$f"; done && rm *.tar.gz
RUN mkdir -p /home/ScienceTools \
    && curl -o /home/ScienceTools/ScienceTools-11-04-00-user.tar.gz ${SLAC_ST_URL}/11-04-00/ScienceTools-11-04-00-user.tar.gz \
    && cd /home/ScienceTools && tar xzf ScienceTools-11-04-00-user.tar.gz && rm ScienceTools-11-04-00-user.tar.gz
