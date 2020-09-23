from preprocessing import preprocess
from compoundFeaturization.deepChemFeatsGenerator import DeepChemFeaturizerGenerator
from featureSelection.featureSelection import featureSelection
import deepchem as dc
import numpy as np
from deepchem.models import GraphConvModel, MultitaskClassifier, RobustMultitaskClassifier

#TODO: define some args as *kwargs and do something by default except if the args are defined ???
dataset_path = preprocess(path='data/dataset_last_version2.csv', smiles_header='Smiles', sep=';', header=0, n=1000)[1]

#featurizer = dc.feat.ConvMolFeaturizer()
featurizer = dc.feat.CircularFingerprint(radius = 2, size = 2048)

loader = dc.data.CSVLoader(
      tasks=["Class"], smiles_field="Canonical_Smiles",
      featurizer=featurizer)

dataset = loader.featurize(dataset_path)

splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)

print(train_dataset)

#transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]

metric = dc.metrics.Metric(dc.metrics.accuracy_score, np.mean)

#model = GraphConvModel(1, batch_size=128, mode='classification')
model = RobustMultitaskClassifier(1, 2048)

print(type(train_dataset))
# Fit trained model
model.fit(train_dataset, nb_epoch=20)
'''
print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

print("Test scores")
print(test_scores)
'''

'''
#TODO: define some args as *kwargs and do something by default except if the args are defined ???
rdkit_fps = DeepChemFeaturizerGenerator(dataset, 'Smiles', 'Class', 'rdkit')
rdkit_dataset = rdkit_fps.getFeaturizerDataset()

#TODO: define some args as *kwargs and do something by default except if the args are defined ???
fs = featureSelection(rdkit_dataset, 'Smiles', 'Class', 'selectFromModel', 0.002)
final_dataset = fs.get_fsDataset()
c_indexes = fs.column_indexes
print(len(c_indexes), rdkit_dataset.shape, final_dataset.shape)
'''





