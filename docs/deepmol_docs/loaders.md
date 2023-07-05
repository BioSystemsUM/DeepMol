# Loading data into DeepMol

It is possible to read data from CSV and SDF files or directly from numpy arrays / lists.

## Using CSVLoader

The CSVLoader class can be used to load tabular data from files. It accepts the following arguments:
- dataset_path: path to the CSV file (mandatory)
- smiles_field: name of the field with the SMILES strings (mandatory)
- id_field: name of the field with the molecules' identifiers (optional, if not provided, an arbitrary, unique identifier will be assigned to each molecule)
- labels_fields: list with the name(s) of the field(s) of the label(s) (optional)
- features_fields: list with the name(s) of the field(s) of the feature(s) (optional)
- shard_size: if you don't want to load the entire data you can define a number of rows to read (optional)
- mode: mode of the dataset, it can be 'classification', 'regression',
and list of tasks for multitask ML, e.g. ['classification', 'classification'] 
when the two labels are for classification. 'auto' will try to automatically infer the mode. 
(optional)

To create the dataset you can use the create_dataset method. It accepts the same arguments as pandas.read_csv().


```python
from deepmol.loaders import CSVLoader

# Load data from CSV file
loader = CSVLoader(dataset_path='../data/example_data_with_features.csv',
                   smiles_field='mols',
                   id_field='ids',
                   labels_fields=['y'],
                   features_fields=[f'feat_{i+1}' for i in range(1024)],
                   shard_size=500,
                   mode='auto')
# create the dataset
csv_dataset = loader.create_dataset(sep=',', header=0)
```

    2023-05-24 16:51:56,426 — INFO — Assuming classification since there are less than 10 unique y values. If otherwise, explicitly set the mode to 'regression'!


## Using SDFLoader

SDF stands for "Structural Data File," which is a file format commonly used in chemistry and bioinformatics to represent the structure of molecules.

SDF files typically contain information such as the atom and bond types, atom positions, and various molecular properties. These files can be used for a variety of purposes, including chemical database storage, chemical structure searching, and molecular modeling.

The SDFLoader class can be used to load data from SDF files. It accepts the following arguments:
- dataset_path: path to the SDF file (mandatory)
- id_field: name of the field with the molecules' identifier (optional, if not provided, an arbitrary, unique identifier will be assigned to each molecule)
- labels_fields: list with the name(s) of the field(s) of the label(s) (optional)
- features_fields: list with the name(s) of the field(s) of the feature(s) (optional)
- shard_size: if you don't want to load the entire data you can define a number of rows to read (optional)
- mode: mode of the dataset, it can be 'classification', 'regression' and list of 
tasks for multitask ML, e.g. ['classification', 'classification'] when the two labels are for classification. 'auto' will try to automatically infer the mode. (optional)

To create the dataset you can use the create_dataset method.

```python
from deepmol.loaders import SDFLoader

# Load data from SDF file
loader = SDFLoader(dataset_path='../data/example_sdf_file.sdf',
                   id_field='ChEMBL_ID',
                   labels_fields=['pIC50'],
                   features_fields=None,
                   shard_size=500,
                   mode='auto')
# create the dataset
sdf_dataset = loader.create_dataset()
```

## Directly from numpy arrays / lists

Directly from numpy arrays / lists as a SmilesDataset (both CSVLoader and SDFLoader return SmilesDataset objects).

A SmilesDataset can be initialized with SMILES strings or RDKit molecules (through the from_mols class method). It accepts the following arguments:
- smiles: list of SMILES strings (mandatory)
- mols: list of RDKit molecules (optional)
- ids: list of molecules' identifiers (optional, if not provided, an arbitrary, unique identifier will be assigned to each molecule)
- X: numpy array with the features (optional)
- feature_names: numpy array of feature names (optional)
- y: numpy array with the labels (optional)
- label_names: numpy array of label names (optional)
- mode: mode of the dataset, it can be 'classification', 'regression', and list of tasks for multitask ML, 
e.g. ['classification', 'classification'] when the two labels are for classification. 'auto' will try to automatically infer the mode. (optional)

In the case of using the from_mols class method, the smiles argument is not used and the mols argument is mandatory.


```python
from rdkit import Chem
from deepmol.datasets import SmilesDataset

smiles = ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C  ', 'CC(=O)OC1=CC=CC=C1C(=O)O']
ids = ['caffeine', 'aspirin']
# x = np.array([[1, 2, 3], [4, 5, 6]])
# ...

df_smiles = SmilesDataset(smiles=smiles, # only mandatory argument, a list of SMILES strings
                          mols=None,
                          ids=ids,
                          X=None,
                          feature_names=None,
                          y=None,
                          label_names=None,
                          mode='auto')

mols = [Chem.MolFromSmiles(s) for s in smiles]
df_mols = SmilesDataset.from_mols(mols=mols, # only mandatory argument, a list of RDKit molecules
                                  ids=ids,
                                  X=None,
                                  feature_names=None,
                                  y=None,
                                  label_names=None,
                                  mode='auto')
```

## Access the data stored in the datasets

- dataset.smiles: list of SMILES strings
- dataset.mols: list of RDKit molecules
- dataset.ids: list of molecules' ids
- dataset.X: numpy array with the features
- dataset.y: numpy array with the labels
- dataset.feature_names: numpy array of feature names
- dataset.label_names: numpy array of label names
- dataset.mode: mode of the dataset, it can be 'classification', 'regression' and list of tasks for multitask ML, 
e.g. ['classification', 'classification'] when the two labels are for classification.
- dataset.n_tasks: number of tasks


```python
from deepmol.loaders import CSVLoader
# Load data from CSV file
loader = CSVLoader(dataset_path='../data/example_data_with_features.csv',
                   smiles_field='mols',
                   id_field='ids',
                   labels_fields=['y'],
                   features_fields=[f'feat_{i+1}' for i in range(1024)],
                   shard_size=500,
                   mode='auto')
# create the dataset
csv_dataset = loader.create_dataset(sep=',', header=0)
```


```python
csv_dataset.smiles[:5]
```




    array(['CC1=CC=C(CC(C)NC(=O)C(N)CC(=O)O)C=C1',
           'CC(C)C(N)C(=O)NC(C(=O)NC(CC1=CC=CC=C1)C(=O)NC(CC(N)=O)C(=O)O)C(C)C',
           'CCSC(=O)CN=C(C)O',
           'CCC(C)CCCCCCCCCCC(=O)OC[C@@H](COC(=O)CCCCCCCCCCC(C)C)OC(=O)CCCCCCCCCCCCCCC(C)C',
           'OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O'], dtype=object)




```python
csv_dataset.mols[:5]
```




    array([<rdkit.Chem.rdchem.Mol object at 0x7ffac7a2ff20>,
           <rdkit.Chem.rdchem.Mol object at 0x7ffac7a2fcf0>,
           <rdkit.Chem.rdchem.Mol object at 0x7ffac7a2fc80>,
           <rdkit.Chem.rdchem.Mol object at 0x7ffac7a2fc10>,
           <rdkit.Chem.rdchem.Mol object at 0x7ffac7a2fb30>], dtype=object)




```python
csv_dataset.ids[:5]
```




    array(['d38fa87cb28c43699734ac0291af33a3',
           '253182410b4145b0967b2e8d10b519d0',
           '36e777f49ba649c3830e161b03bbb777',
           'ba6f25d48cf94562b4e0485cf0b23aef',
           'e524254dd83a4206ba61e526862f6da2'], dtype='<U32')




```python
csv_dataset.X[:5]
```




    array([[0., 1., 0., ..., 0., 0., 0.],
           [0., 1., 0., ..., 0., 1., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 1., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
csv_dataset.y[:5]
```




    array([1., 1., 0., 0., 1.])




```python
csv_dataset.feature_names[:5]
```




    array(['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5'], dtype='<U9')




```python
csv_dataset.label_names
```




    array(['y'], dtype='<U1')




```python
csv_dataset.mode
```




    'classification'




```python
csv_dataset.n_tasks
```




    1


