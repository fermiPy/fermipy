#echo 'DOCKER_OPTS="-H tcp://127.0.0.1:2375 -H unix:///var/run/docker.sock -s devicemapper"' | sudo tee /etc/default/docker > /dev/null
#sudo service docker restart
docker build -t mdwood/fermist .
docker info
docker images
docker run -it -d -v $PWD:/home/fermipy --name=test0 mdwood/fermist /bin/bash
docker exec test0 /bin/bash -c "cd /home/fermipy;/bin/bash /home/travisinstall.sh"
