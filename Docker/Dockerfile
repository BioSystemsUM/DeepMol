FROM ubuntu:focal

RUN adduser deepmol
USER deepmol
WORKDIR /home

ENV HOME_DIR="/home/deepmol"

ENV PATH="/home/deepmol/.local/bin:${PATH}"

ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=deepmol \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=$HOME_DIR

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
RUN apt-get update

RUN apt-get install -y git wget && \
    apt-get install --no-install-recommends --yes build-essential ffmpeg libsm6 libxext6

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
#    echo "bfe34e1fa28d6d75a7ad05fd02fa5472275673d5f5621b77380898dee1be15d2 *Miniconda3-4.7.12.1-Linux-x86_64.sh" | sha256sum -c - && \
RUN /bin/bash /home/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py310_23.5.2-0-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

RUN conda config --add pkgs_dirs $HOME_DIR


ARG python_version=3.10

#RUN apt-get install build-essential swig

RUN pip install deepmol[all]

RUN pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

