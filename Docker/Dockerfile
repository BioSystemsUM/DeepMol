

FROM tensorflow/tensorflow:2.4.1-gpu

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

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
    echo "bfe34e1fa28d6d75a7ad05fd02fa5472275673d5f5621b77380898dee1be15d2 *Miniconda3-4.7.12.1-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.7.12.1-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.7.12.1-Linux-x86_64.sh && \
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
      joblib \
      pillow \
      #tensorflow-gpu==2.4.1 \
      h5py \
      keras==2.4.0 \
      plotly \
      seaborn \
      kneed \
      deepchem==2.5.0 \
      && \
    conda install \
      scikit-learn==0.23.2 \
      chembl_structure_pipeline \
      matplotlib \
      notebook \
      pandas \
      pydot \
      pyyaml \
      rdkit==2021.03.1 \
      tqdm \
      joblib \
      imbalanced-learn \
      shap==0.39.0 \
      && \
    conda clean -yt


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /data3

EXPOSE 8882

CMD jupyter notebook --port=8882 --ip=0.0.0.0
