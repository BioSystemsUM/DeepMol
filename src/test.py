from preprocessing import preprocess
from compoundFeaturization.deepChemFeatsGenerator import DeepChemFeaturizerGenerator
from featureSelection.featureSelection import featureSelection

dataset = preprocess(path='data/dataset_last_version2.csv', smiles_header='Smiles', sep=';', header=0, n=10)

rdkit_fps = DeepChemFeaturizerGenerator(dataset, 'Smiles', 'Class', 'rdkit')
rdkit_dataset = rdkit_fps.getFeaturizerDataset()

fs = featureSelection(rdkit_dataset, 'Smiles', 'Class', 'selectFromModel', 0.002)
final_dataset = fs.get_fsDataset()
c_indexes = fs.column_indexes
print(len(c_indexes), rdkit_dataset.shape, final_dataset.shape)



