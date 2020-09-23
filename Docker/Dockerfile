#ARG cuda_version=10.0
#ARG cudnn_version=7
#FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel
FROM tensorflow/tensorflow:1.14.0-gpu-py3
#FROM tensorflow/tensorflow:latest-gpu-py3

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh


# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

USER $NB_USER

ARG python_version=3.6

#RUN apt-get install build-essential swig

RUN conda config --append channels conda-forge
RUN conda install -y python=${python_version} && \
    pip install --upgrade pip --ignore-installed six --upgrade cython && \
    pip install \
#      sklearn_pandas \
      joblib \
      pillow \
      mordred \
      molvs \
      deepchem \
      git+https://github.com/samoturk/mol2vec \
      tensorflow-gpu==1.14 \
#      setuptools \
#      lockfile \
#      pyyaml \
#      ConfigSpace \
#      pynisher \
#      pyrfr \
#      smac \
#      auto-sklearn \
      cntk-gpu && \
    conda install \
      scikit-learn==0.22.0 \
#      rdkit deepchem \
      bcolz \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pydot \
      pygpu \
      pyyaml \
      six \
      theano \
      mkdocs \
      rdkit \
      tqdm \
      joblib \
      bunch \
      gensim \
      imbalanced-learn \
      && \
    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git && \
    conda clean -yt

ADD theanorc /home/keras/.theanorc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /data3

EXPOSE 8882

CMD jupyter notebook --port=8882 --ip=0.0.0.0
