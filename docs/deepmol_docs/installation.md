# Installation

## Pip

Install DeepMol via pip:

If you intend to install all the deepmol modules' dependencies:

```bash
pip install deepmol[all]
```

or in MacOS:

```bash
pip install "deepmol[all]"
```


Extra modules:

```bash
pip install deepmol[preprocessing]
pip install deepmol[machine-learning]
pip install deepmol[deep-learning]
```

or in MacOS:

```bash
pip install "deepmol[preprocessing]"
pip install "deepmol[machine-learning]"
pip install "deepmol[deep-learning]"
```

Also, you should install mol2vec and its dependencies:

```bash
pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec
```

## Manually

Alternatively, clone the repository and install the dependencies manually:

1. Clone the repository:
```bash
git clone https://github.com/BioSystemsUM/DeepMol.git
```

3. Install dependencies:
```bash
python setup.py install
```

## Docker

You can also use the provided image to build your own Docker image:

```bash
docker pull biosystemsum/deepmol
```

## Disclaimer

If you’d like to use the GPU, make sure to install the versions of TensorFlow and DGL that match the CUDA drivers for your hardware.

Do not install JAX, it will result dependency conflicts. 

Loading tensorflow models will be problematic for MacOS users due to a known tensorflow issue [46](https://github.com/keras-team/tf-keras/issues/46).