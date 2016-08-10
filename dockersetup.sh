docker build -t mdwood/fermist .
docker info
docker images
#docker run -it mdwood/fermist /bin/bash
docker run -d -p 127.0.0.1:80:80 mdwood/fermist
ls -l
