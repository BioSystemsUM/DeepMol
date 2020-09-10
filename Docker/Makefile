help:
	@cat Makefile

DATA?="${HOME}/PycharmProjects/DeepMol/src"
#GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=docker
#GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
#CUDA_VERSION?=9.0
#CUDNN_VERSION?=7
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
	docker build -t keras --build-arg python_version=$(PYTHON_VERSION) -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run --name DockerDeepMol --gpus all -it --rm --user 1000:1000 -v $(SRC):/src/workspace -v $(DATA):/data3 --net=host --env KERAS_BACKEND=$(BACKEND) keras bash

ipython: build
	$(DOCKER) run --name DockerDeepMol --gpus all -it --rm --user 1000:1000 -v $(SRC):/src/workspace -v $(DATA):/data3 --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	$(DOCKER) run --name DockerDeepMol --gpus all -it --rm --user 1000:1000 -v $(SRC):/src/workspace -v $(DATA):/data3 --net=host --env KERAS_BACKEND=$(BACKEND) keras

test: build
	$(DOCKER) run --name DockerDeepMol --gpus all -it --rm --user 1000:1000 -v $(SRC):/src/workspace -v $(DATA):/data3 --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)

