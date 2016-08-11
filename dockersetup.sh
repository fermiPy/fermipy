#echo 'DOCKER_OPTS="-H tcp://127.0.0.1:2375 -H unix:///var/run/docker.sock -s devicemapper"' | sudo tee /etc/default/docker > /dev/null
#sudo service docker restart
docker build -t mdwood/fermist .
docker info
docker images
docker run -it -d -v $PWD:/home/fermipy --name=test0 \
       -e GLAST_EXT=/home/externals \
       -e INST_DIR=/home \
       -e CONDA_DOWNLOAD="$CONDA_DOWNLOAD" \
       -e CONDA_DEPS="$CONDA_DEPS" \
       -e PYTHON_VERSION="$PYTHON_VERSION" \
       -e CONDA2="$CONDA2" \
       mdwood/fermist /bin/bash
docker attach test0
#docker exec test0 /bin/bash -c "cd /home/fermipy;/bin/bash /home/travisinstall.sh"
