from compoundFeaturization.rdkitFingerprints import MorganFingerprint
from Dataset.Dataset import CSVLoader
from featureSelection.baseFeatureSelector import LowVarianceFS
from splitters.splitters import RandomSplitter
from models.sklearnModels import SklearnModel
from metrics.Metrics import Metric
from metrics.metricsFunctions import roc_auc_score, precision_score
from parameterOptimization.HyperparameterOpt import GridHyperparamOpt


#TODO: try with chunks

ds = CSVLoader('preprocessed_dataset.csv', 'Smiles', ['Class'], 'PubChem CID', chunk_size=55)
ds = MorganFingerprint().featurize(ds)

#print(ds.X)
#print(ds.y)
#print(ds.features)
#print(ds.get_shape())

ds = LowVarianceFS(0.15).featureSelection(ds)


splitter = RandomSplitter()

train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6, frac_valid=0.2, frac_test=0.2)

#print(train_dataset.X)
#print(train_dataset.y)
#print(train_dataset.ids)
#print(train_dataset.features)
#print(train_dataset.features2keep)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#TODO: deal with metrics that do not accept predict_proba output (ex: in rf)
rf = RandomForestClassifier()
svm = SVC()

model = SklearnModel(model=svm)
# model training
model.fit(train_dataset)

valid_preds = model.predict(valid_dataset)
test_preds = model.predict(test_dataset)


#TODO: Chech the problem with the metrics
metrics = [Metric(roc_auc_score), Metric(precision_score)]
# evaluate the model
print('Training Dataset: ')
train_score = model.evaluate(train_dataset, metrics)
print('Validation Dataset: ')
valid_score = model.evaluate(valid_dataset, metrics)
print('Test Dataset: ')
test_score = model.evaluate(test_dataset, metrics)



params_dict = {"C": [1.0, 0.7],
               "gamma": ["scale", "auto"],
               "kernel": ["linear", "rbf"]
              }
optimizer = GridHyperparamOpt(model)

best_svm, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset, Metric(precision_score))

print(best_hyperparams)
print(all_results)

