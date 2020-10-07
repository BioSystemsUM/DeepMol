from Dataset.loaders import CSVLoader

ds = CSVLoader('preprocessed_dataset.csv', 'Smiles', ['Class'], 'PubChem CID')

#print(ds.dataset_path, ds.tasks, ds.input_field, ds.id_field, ds.user_features)

print(ds.dataset)