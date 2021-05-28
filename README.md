# DeepMol

DeepMol is a python-based machine and deep learning framework for drug discovery. 
It offers a variety of functionalities that enable a smoother approach to many 
drug discovery and chemoinformatics problems. It uses Tensorflow, Keras, 
Scikit-learn and DeepChem to build custom ML and DL models or 
make use of pre-built ones. It uses the RDKit framework to perform 
operations on molecular data.

### Table of contents:

- [Requirements](#requirements)
- [Installation](#installation)
    - [Pip](#pip)
    - [Docker](#docker)
- [Getting Started](#getting-started)
- [About Us](#about-us)
- [Citing DeepMol](#citing-deepmol)

## Requirements


## Installation

### Pip

### Docker


## Getting Started

```python
from loaders.Loaders import CSVLoader

# load a dataset from a CSV (define data path, field with the molecules,
# field with the labels (optional), field with ids (optional), etc.
dataset = CSVLoader(dataset_path='data_path.csv', 
                    mols_field='Smiles', 
                    labels_fields='Class', 
                    id_field='ID')
dataset = dataset.create_dataset()

#print shape of the dataset (mols, X, y) 
dataset.get_shape()
```

<p align="left">
  <img src="" width="800" />
</p>

## About Us

DeepMol is managed by a team of contributors from the BioSystems group 
at the Centre of Biological Engineering, University of Minho.

## Citing DeepMol

Manuscript under preparation.