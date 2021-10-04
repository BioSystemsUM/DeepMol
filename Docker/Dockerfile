#FROM tensorflow/tensorflow:2.2.1-gpu-py3
#FROM tensorflow/tensorflow:2.2.2-gpu-py3
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04
# FROM tensorflow/tensorflow:2.4.1-gpu

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

#RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh && \
#    echo "bfe34e1fa28d6d75a7ad05fd02fa5472275673d5f5621b77380898dee1be15d2 *Miniconda3-4.7.12.1-Linux-x86_64.sh" | sha256sum -c - && \
#    /bin/bash /Miniconda3-4.7.12.1-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
#    rm Miniconda3-4.7.12.1-Linux-x86_64.sh && \
#    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
#    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
#    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
#    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
#    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA

ENV PATH /miniconda/bin:$PATH


# Install Python packages and keras
#ENV NB_USER keras
#ENV NB_UID 1000

#RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
#    chown $NB_USER $CONDA_DIR -R && \
#    mkdir -p /src && \
#    chown $NB_USER /src

# USER $NB_USER

ARG python_version=3.6

# RUN conda config --append channels conda-forge
RUN conda update -n base -c defaults conda -y
RUN conda install -y -c anaconda python=${python_version}
RUN conda install -y -c anaconda pandas && conda clean -afy
RUN conda install -y -c anaconda matplotlib && conda clean -afy
RUN conda install -y -c anaconda seaborn && conda clean -afy
RUN conda install -y -c plotly plotly && conda clean -afy
RUN conda install -y -c conda-forge kneed && conda clean -afy
RUN conda install -y -c anaconda scikit-learn=0.23.2 && conda clean -afy
RUN conda install -y -c anaconda tensorflow-gpu=2.2.0 && conda clean -afy
#RUN conda install -y -c conda-forge cudatoolkit=11.1.1 && conda clean -afy
#RUN conda install -y -c anaconda tensorflow-gpu=2.4.1 && conda clean -afy
RUN conda install -y -c conda-forge keras && conda clean -afy
RUN conda install -y -c pytorch pytorch=1.6.0 && conda clean -afy
RUN conda install -y -c conda-forge pytorch_geometric && conda clean -afy
RUN conda install -y -c rdkit rdkit && conda clean -afy
RUN conda install -y -c mcs07 pubchempy && conda clean -afy
RUN conda install -y -c rdkit -c mordred-descriptor mordred && conda clean -afy
RUN conda install -y -c dglteam dgl-cuda11.1 && conda clean -afy
RUN conda install -y -c dglteam dgllife && conda clean -afy
RUN conda install -y -c conda-forge deepchem=2.5.0 && conda clean -afy
RUN conda install -y -c mordred-descriptor mordred && conda clean -afy
RUN conda install -y -c conda-forge imbalanced-learn && conda clean -afy
RUN conda install -y -c conda-forge molvs && conda clean -afy
RUN conda install -y -c conda-forge chembl_structure_pipeline && conda clean -afy
RUN conda install -y -c conda-forge theano && conda clean -afy
RUN conda install -y -c anaconda notebook && conda clean -afy
RUN conda install -y -c conda-forge mkdocs && conda clean -afy
RUN conda install -y -c anaconda gensim=3.8.3 && conda clean -afy
RUN conda install -y -c conda-forge shap && conda clean -afy
RUN python3 -m pip -q --no-cache-dir install --upgrade pip
RUN python3 -m pip -q --no-cache-dir install git+https://github.com/samoturk/mol2vec
RUN python3 -m pip -q --no-cache-dir install cntk-gpu
RUN python3 -m pip -q --no-cache-dir install tensorflow-probability==0.10.1
RUN python3 -m pip -q --no-cache-dir install git+https://github.com/BioSystemsUM/DeepMol.git@pacbb21

ADD /Docker/theanorc /home/keras/.theanorc

ADD /pacbb21_paper /home/pacbb21_paper
WORKDIR /home/pacbb21_paper

# RUN git clone https://github.com/BioSystemsUM/DeepMol.git --branch pacbb21
# WORKDIR /DeepMol
# RUN python3 -m pip -q --no-cache-dir install .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 8882

CMD jupyter notebook --port=8882 --ip=0.0.0.0
