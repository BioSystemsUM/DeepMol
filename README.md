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
    - [Compound Standardization](#compound-standardization)
    - [Compound Featurization](#compound-featurization)
    - [Feature Selection](#feature-selection)
    - [Unsupervised Exploration](#unsupervised-exploration)
    - [Data Split](#data-split)
    - [Build, train and evaluate a model](#build-train-and-evaluate-a-model)
    - [Hyperparameter Optimization](#hyperparameter-optimization)
    - [Feature Importance (Shap Values)](#feature-importance-shap-values)
    - [Unbalanced Datasets](#unbalanced-datasets)
- [About Us](#about-us)
- [Citing DeepMol](#citing-deepmol)

## Requirements


## Installation

### Pip

### Docker


## Getting Started

DeepMol is built in a modular way allowing the use of its methods for 
multiple tasks. ...


### Load a dataset from a CSV

For now it is only possible to load data directly from CSV files. Modules to 
load data from different file types and sources will be implemented in the 
future. These include from JSON, SDF and FASTA files and directly from our 
databases.

To load data from a CSV its only required to provide the math and molecules 
field name. Optionaly it is also possible to provide a field with some ids, 
the labels fields, features fields, features to keep (usefull for instance 
to select only the features kept after feature selection)

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

### Compound Standardization

bla bla

### Compound Featurization

morgan, mol2vec, etc

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

LowVariance, FromModel, ...

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

### Unsupervised Exploration

PCA, tSNE, Kmeans, UMAP ...

```python
from unsupervised.umap import UMAP

ump = UMAP().runUnsupervised(dataset)
```

<p align="left">
  <img src="https://raw.githubusercontent.com/BioSystemsUM/DeepMol/master/src/docs/imgs/umap_output.png?token=AGEFRGMRHLMDFL7MJEJEVVDAXXW5G" width="800" />
</p>

### Data Split

bla bla 

```python
from splitters.splitters import SingletaskStratifiedSplitter

#Data Split
splitter = SingletaskStratifiedSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset, frac_train=0.7, 
                                                                             frac_valid=0.15, frac_test=0.15)
print('Train:')
train_dataset.get_shape()
print('\nValidation:')
valid_dataset.get_shape()
print('\nTest:')
test_dataset.get_shape()
```

<p align="left">
  <img src="https://raw.githubusercontent.com/BioSystemsUM/DeepMol/master/src/docs/imgs/split_output.png?token=AGEFRGJQOLD56IUM4Z3JEI3AXXXAO" width="800" />
</p>

### Build, train and evaluate a model

bla bla

#### Scikit-Learn model example

Full jupyter notebook here! (put link)

```python
from sklearn.ensemble import RandomForestClassifier
from models.sklearnModels import SklearnModel

#Scikit-Learn Random Forest
rf = RandomForestClassifier()
#wrapper around scikit learn models
model = SklearnModel(model=rf)
# model training
model.fit(train_dataset)

from metrics.Metrics import Metric
from metrics.metricsFunctions import r2_score, roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report, f1_score


#cross validate model on the full dataset
model.cross_validate(dataset, Metric(roc_auc_score), folds=3)
```

<p align="left">
  <img src="https://raw.githubusercontent.com/BioSystemsUM/DeepMol/master/src/docs/imgs/cross_validation_output.png?token=AGEFRGMVOC546LOEGLVEZP3AXXXDA" width="800" />
</p>

```python
#evaluate the model using different metrics
metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix), 
           Metric(classification_report)]

# evaluate the model on trining data
print('Training Dataset: ')
train_score = model.evaluate(train_dataset, metrics)

# evaluate the model on trining data
print('Validation Dataset: ')
valid_score = model.evaluate(valid_dataset, metrics)

# evaluate the model on trining data
print('Test Dataset: ')
test_score = model.evaluate(test_dataset, metrics)
```

<p align="left">
  <img src="https://raw.githubusercontent.com/BioSystemsUM/DeepMol/master/src/docs/imgs/evaluate_output.png?token=AGEFRGPO25CGSP6Y73N5OUTAXXXI4" width="800" />
</p>

#### Keras model example

Full jupyter notebook here! (put link)

```python
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np


input_dim = train_dataset.X.shape[1]


def create_model(optimizer='adam', dropout=0.5, input_dim=input_dim):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

from models.kerasModels import KerasModel
model = KerasModel(create_model, epochs = 5, verbose=1, optimizer='adam')

#train model
model.fit(train_dataset)

#make prediction on the test dataset with the model
model.predict(test_dataset)

#evaluate model using multiple metrics
metrics = [Metric(roc_auc_score), 
           Metric(precision_score), 
           Metric(accuracy_score), 
           Metric(confusion_matrix), 
           Metric(classification_report)]


print('Training set score:', model.evaluate(train_dataset, metrics))
print('Test set score:', model.evaluate(test_dataset, metrics))
```


#### DeepChem model example

Full jupyter notebook here! (put link)

...

### Hyperparameter Optimization

bla bla

```python
from parameterOptimization.HyperparameterOpt import HyperparamOpt_Valid, HyperparamOpt_CV
#Hyperparameter Optimization (using the above created keras model)
optimizer = HyperparamOpt_Valid(create_model)

params_dict = {'optimizer' : ['adam', 'rmsprop'],
              'dropout' : [0.2, 0.4, 0.5]}

best_model, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict, train_dataset, 
                                                                        valid_dataset, Metric(roc_auc_score))


print(best_hyperparams)
print(best_model)

#Evaluate model
best_model.evaluate(test_dataset, metrics)
```

### Feature Importance (Shap Values)

```python
from unsupervised.umap import UMAP

ump = UMAP().runUnsupervised(dataset)
```

### Unbalanced Datasets

```python
from 


```



## About Us

DeepMol is managed by a team of contributors from the BioSystems group 
at the Centre of Biological Engineering, University of Minho.

## Citing DeepMol

Manuscript under preparation.