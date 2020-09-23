from preprocessing import preprocess
from compoundFeaturization.deepChemFeatsGenerator import DeepChemFeaturizerGenerator
from featureSelection.featureSelection import featureSelection
import deepchem as dc
import numpy as np
from deepchem.models import GraphConvModel, MultitaskClassifier, RobustMultitaskClassifier

#TODO: define some args as *kwargs and do something by default except if the args are defined ???
dataset_path = preprocess(path='data/dataset_last_version2.csv', smiles_header='Smiles', sep=';', header=0, n=1000)[1]

featurizer = dc.feat.ConvMolFeaturizer()
#featurizer = dc.feat.CircularFingerprint(radius = 2, size = 2048)

loader = dc.data.CSVLoader(
      tasks=["Class"], smiles_field="Canonical_Smiles",
      featurizer=featurizer)

dataset = loader.featurize(dataset_path)

splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)

print(train_dataset)

transformers = [dc.trans.BalancingTransformer(dataset=train_dataset)]

for transformer in transformers:
      train_dataset = transformer.transform(train_dataset)
      valid_dataset = transformer.transform(valid_dataset)
      test_dataset = transformer.transform(test_dataset)

metric = dc.metrics.Metric(dc.metrics.accuracy_score, np.mean)

model = GraphConvModel(1, batch_size=128, mode='classification')
#model = RobustMultitaskClassifier(1, 2048)

print(type(train_dataset))
# Fit trained model
model.fit(train_dataset, nb_epoch=20)

metric1 = dc.metrics.Metric(dc.metrics.roc_auc_score)
metric2 = dc.metrics.Metric(dc.metrics.accuracy_score)
metric3 = dc.metrics.Metric(dc.metrics.f1_score)
print('Training set score:', model.evaluate(train_dataset, [metric1, metric2, metric3], transformers))
print('Test set score:', model.evaluate(test_dataset, [metric1, metric2, metric3], transformers))

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





