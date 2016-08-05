FROM mdwood/fermist:11-03-00-v0
MAINTAINER Matthew Wood <mdwood@slac.stanford.edu>
RUN yum install -y git
RUN yum install -y libgomp
RUN yum install -y libXext
RUN yum install -y libSM
RUN yum install -y libXrender
