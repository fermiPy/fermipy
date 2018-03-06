#echo 'DOCKER_OPTS="-H tcp://127.0.0.1:2375 -H unix:///var/run/docker.sock -s devicemapper"' | sudo tee /etc/default/docker > /dev/null
#sudo service docker restart
cat $1/Dockerfile
docker build -t fermist \
       --build-arg PIP_DEPS="$PIP_DEPS" \
       --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
       --build-arg CONDA_DEPS="$CONDA_DEPS" \
       --build-arg CONDA_DOWNLOAD="$CONDA_DOWNLOAD" \
       $1
docker info
docker images
docker run -it -d -v $PWD:/home/fermipy --tmpfs /tmp --name=fermipy-testing \
       -e TRAVIS="$TRAVIS" \
       -e TRAVIS_JOB_ID="$TRAVIS_JOB_ID" \
       -e TRAVIS_BRANCH="$TRAVIS_BRANCH" \
       fermist /bin/bash

