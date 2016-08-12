FROM mdwood/fermist:11-03-00-v0
MAINTAINER Matthew Wood <mdwood@slac.stanford.edu>
RUN yum install -y git libgomp libXext libSM libXrender libicu xorg-x11-server-Xvfb
RUN cd /home; tar xzf ScienceTools-11-03-00-user.tar.gz
RUN cd /home/externals; for f in *.tar.gz; do tar -xf "$f"; done
