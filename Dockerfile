FROM mdwood/fermist:11-03-00-v0
MAINTAINER Matthew Wood <mdwood@slac.stanford.edu>
RUN yum install -y git libgomp libXext libSM libXrender
#ADD dockerinstall.sh
