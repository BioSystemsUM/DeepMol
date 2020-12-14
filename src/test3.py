from compoundFeaturization.rdkitFingerprints import MorganFingerprint, MACCSkeysFingerprint, LayeredFingerprint
from compoundFeaturization.rdkitFingerprints import RDKFingerprint, AtomPairFingerprint
from compoundFeaturization.mol2vec import Mol2Vec
from Dataset.Dataset import CSVLoader, NumpyDataset
from featureSelection.baseFeatureSelector import LowVarianceFS, KbestFS, PercentilFS, RFECVFS, SelectFromModelFS
from splitters.splitters import RandomSplitter
from models.sklearnModels import SklearnModel
from metrics.Metrics import Metric
from metrics.metricsFunctions import roc_auc_score, precision_score, accuracy_score
from parameterOptimization.HyperparameterOpt import GridHyperparamOpt
import preprocessing as preproc
from imbalanced_learn.ImbalancedLearn import RandomOverSampler

#pp_ds, path = preproc.preprocess(path='data/dataset_last_version2.csv', smiles_header='Smiles', sep=';', header=0, n=None)
#pp_ds, path = preproc.preprocess(path='data/datset_wFooDB.csv',
#                                 smiles_header='SMILES',
#                                 class_header='sweet',
#                                 ids_header='compound id',
#                                 sep='\t',
#                                 header=0,
#                                 n=None,
#                                 save_path='preprocessed_dataset_wfoodb.csv')

#ds = NumpyDataset(X=pp_ds.Standardized_Smiles, y=pp_ds.Class)

#ds = CSVLoader('preprocessed_dataset.csv', 'Smiles', ['Class'], 'PubChem CID')#, chunk_size=1000)

ds = CSVLoader('preprocessed_dataset_wfoodb.csv', 'Smiles', ['Class'], 'ID')#, chunk_size=1000)

ds.get_shape()

ds = MorganFingerprint().featurize(ds)
#ds = MACCSkeysFingerprint().featurize(ds)
#ds = LayeredFingerprint().featurize(ds)
#ds = RDKFingerprint().featurize(ds)
#ds = AtomPairFingerprint().featurize(ds)
#ds = Mol2Vec().featurize(ds)

#print(ds.X)
#print(ds.y)
#print(ds.features)
#print(ds.get_shape())

print('-----------------------------------------------------')
ds.get_shape()

ds = LowVarianceFS(0.15).featureSelection(ds)
#ds = KbestFS().featureSelection(ds)
#ds = PercentilFS().featureSelection(ds)
#ds = RFECVFS().featureSelection(ds)
#ds = SelectFromModelFS().featureSelection(ds)

ds.get_shape()

splitter = RandomSplitter()

train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6, frac_valid=0.2, frac_test=0.2)

train_dataset = RandomOverSampler().sample(train_dataset)


#k_folds = splitter.k_fold_split(ds, 3)

#for a, b in k_folds:
#    print(a.get_shape())
#    print(b.get_shape())
#    print('############')


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

model = SklearnModel(model=rf)

#print(model.cross_validate(ds, Metric(roc_auc_score)))


# model training
model.fit(train_dataset)

valid_preds = model.predict(valid_dataset)
test_preds = model.predict(test_dataset)


metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
# evaluate the model
#print('Training Dataset: ')
train_score = model.evaluate(train_dataset, metrics)
#print('Validation Dataset: ')
valid_score = model.evaluate(valid_dataset, metrics)
#print('Test Dataset: ')
test_score = model.evaluate(test_dataset, metrics)

def rf_model_builder(n_estimators, max_features, class_weight, model_dir=None):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, class_weight=class_weight)
    return SklearnModel(rf_model, model_dir)

params_dict_rf = {"n_estimators": [10, 100],
                  "max_features": ["auto", "sqrt", "log2", None],
                  "class_weight": [{0: 1., 1: 1.}, {0: 1., 1: 5}, {0: 1., 1: 10}]
                  }

def svm_model_builder(C, gamma, kernel, model_dir=None):
    svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
    return SklearnModel(svm_model, model_dir)

params_dict_svm = {'C': [1.0, 0.7, 0.5, 0.3, 0.1],
               'gamma': ["scale", "auto"],
               'kernel': ["linear", "rbf"]
              }

optimizer = GridHyperparamOpt(rf_model_builder)

best_rf, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict_rf, train_dataset, valid_dataset, Metric(roc_auc_score))

print('#################')
print(best_hyperparams)
print(best_rf)

#print(best_rf.predict(test_dataset))
print('@@@@@@@@@@@@@@@@')
print(best_rf.evaluate(test_dataset, metrics))

print(best_rf.predict(test_dataset))
