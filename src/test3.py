from compoundFeaturization.rdkitFingerprints import MorganFingerprint
from Dataset.Dataset import CSVLoader
from featureSelection.baseFeatureSelector import LowVarianceFS

#TODO: try with chunks

ds = CSVLoader('preprocessed_dataset.csv', 'Smiles', ['Class'], 'PubChem CID', chunk_size=100)
ds = MorganFingerprint().featurize(ds)

print(ds.get_shape()) #TODO: resolve shape problem when fps cannot be created for a certain smile

#TODO: finnish this class
f = LowVarianceFS(0.15).featureSelection(ds)





#print(ds.features)

#print(ds.dataset_path, ds.tasks, ds.input_field, ds.id_field, ds.user_features)

#print(ds.dataset)