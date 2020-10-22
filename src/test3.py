from Dataset.loaders import CSVLoader

#TODO: try with chunks
ds = CSVLoader('preprocessed_dataset.csv', 'Smiles', ['Class'], 'PubChem CID', chunk_size=100)

#print(ds.dataset_path, ds.tasks, ds.input_field, ds.id_field, ds.user_features)

print(ds.dataset)