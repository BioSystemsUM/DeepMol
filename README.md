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
    - [Load dataset from csv](#load-a-dataset-from-a-csv)
- [About Us](#about-us)
- [Citing DeepMol](#citing-deepmol)

## Requirements


## Installation

### Pip

### Docker


## Getting Started

### Load a dataset from a CSV
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
  <img src="https://raw.githubusercontent.com/BioSystemsUM/DeepMol/master/src/docs/imgs/load_csv_output.png?token=AGEFRGJBJDLT7RC27OPU5PDAXJRZC" width="800" />
</p>

### Compound Featurization

```python
from compoundFeaturization.rdkitFingerprints import MorganFingerprint

#Compute morgan fingerprints for molecules in the previous loaded dataset
dataset = MorganFingerprint(radius=2, size=1024).featurize(dataset)
```

<p align="left">
  <img src="https://raw.githubusercontent.com/BioSystemsUM/DeepMol/master/src/docs/imgs/featurization_output.png?token=AGEFRGODQDHUOPFRH5NV3P3AXJSZQ" width="800" />
</p>

```python
#print shape of the dataset to see difference in the X shape
dataset.get_shape()
```

<p align="left">
  <img src="https://raw.githubusercontent.com/BioSystemsUM/DeepMol/master/src/docs/imgs/get_shape_output.png?token=AGEFRGOEO6GO33NT7DX3VNLAXJTEE" width="800" />
</p>

### Feature Selection
```python
from featureSelection.baseFeatureSelector import LowVarianceFS

#Feature Selection to remove features with low variance across molecules
dataset = LowVarianceFS(0.15).featureSelection(dataset)

#print shape of the dataset to see difference in the X shape (less features)
dataset.get_shape()
```

<p align="left">
  <img src="https://raw.githubusercontent.com/BioSystemsUM/DeepMol/master/src/docs/imgs/get_shape_output_2.png?token=AGEFRGKWWJ6JAGUYWQ3FQX3AXJUSC" width="800" />
</p>

## About Us

DeepMol is managed by a team of contributors from the BioSystems group 
at the Centre of Biological Engineering, University of Minho.

## Citing DeepMol

Manuscript under preparation.