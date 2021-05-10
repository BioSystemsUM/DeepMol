import os
from src.loaders.Loaders import CSVLoader
from src.splitters.splitters import RandomSplitter, SingletaskStratifiedSplitter
from src.standardizer.ChEMBLStandardizer import ChEMBLStandardizer

# Load Dataset

dataset_paths = ['data/CCRF-CEM.csv', 'data/PC-3.csv', 'data/1-balance.csv', 'data/109-balance.csv']

for dataset_path in dataset_paths:
    if 'CCRF-CEM' in dataset_path or 'PC-3' in dataset_path:
        output_var_name = 'pIC50'
        splitter = RandomSplitter()
    else:
        output_var_name = 'value'
        splitter = SingletaskStratifiedSplitter()

    dataset = CSVLoader(dataset_path=dataset_path,
                        mols_field='smiles',
                        labels_fields=output_var_name)
    dataset = dataset.create_dataset()
    dataset.get_shape()
    # SMILES standardization
    standardizer = ChEMBLStandardizer().standardize(dataset)

    # Data Split
    train_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.7, seed=123)

    dataset_name = os.path.splitext(os.path.split(dataset_path)[-1])[0]
    output_dir = os.path.join('data', 'split_datasets', dataset_name)
    train_dataset.save_to_csv(os.path.join(output_dir, 'train_' + dataset_name + '.csv'))
    test_dataset.save_to_csv(os.path.join(output_dir, 'test_' + dataset_name + '.csv'))
