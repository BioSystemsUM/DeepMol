# Load a dataset

To load data from a CSV it's only required to provide the data path and molecules 
field name. Optionally, it is also possible to provide a field with some ids, 
the labels fields, features fields and the number of  samples to load (by default 
loads the entire dataset).

```python
from deepmol.loaders.loaders import CSVLoader

# load a dataset from a CSV (required fields: dataset_path and smiles_field)
loader = CSVLoader(dataset_path='../../data/train_dataset.csv',
                   smiles_field='mols',
                   id_field='ids',
                   labels_fields=['y'],
                   features_fields=['feat_1', 'feat_2', 'feat_3', 'feat_4'],
                   shard_size=1000,
                   mode='auto')

dataset = loader.create_dataset()

# print shape of the dataset (molecules, X, y)
dataset.get_shape()

((1000,), None, (1000,))
```