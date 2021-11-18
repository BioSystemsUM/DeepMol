from compoundFeaturization.rdkitFingerprints import MorganFingerprint, MACCSkeysFingerprint, LayeredFingerprint
from compoundFeaturization.rdkitFingerprints import RDKFingerprint, AtomPairFingerprint
from compoundFeaturization.mol2vec import Mol2Vec
from loaders.Loaders import CSVLoader
from featureSelection.baseFeatureSelector import LowVarianceFS, KbestFS, PercentilFS, RFECVFS, SelectFromModelFS
from splitters.splitters import SingletaskStratifiedSplitter, RandomSplitter
from models.sklearnModels import SklearnModel
from metrics.Metrics import Metric
from metrics.metricsFunctions import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report
from parameterOptimization.HyperparameterOpt import GridHyperparamOpt
import preprocessing as preproc
from imbalanced_learn.ImbalancedLearn import RandomOverSampler, SMOTEENN
import numpy as np

from standardizer.CustomStandardizer import CustomStandardizer, heavy_standardisation
from unsupervised.baseUnsupervised import PCA


standardizer_vr = CustomStandardizer(params=heavy_standardisation)


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

ds = CSVLoader(dataset_path='tests/data/preprocessed_dataset_wfoodb.csv', mols_field='Smiles', labels_fields='Class', id_field='ID', shard_size=5000)
ds = ds.create_dataset()

ds.get_shape()

ds = MorganFingerprint().featurize(ds)
#ds = MACCSkeysFingerprint().featurize(ds)
#ds = LayeredFingerprint().featurize(ds)
#ds = RDKFingerprint().featurize(ds)
#ds = AtomPairFingerprint().featurize(ds)
#ds = Mol2Vec().featurize(ds)

print(ds.X)
print(ds.y)
print(ds.mols)
print(ds.get_shape())


print('-----------------------------------------------------')

ds = LowVarianceFS(0.15).featureSelection(ds)
#ds = KbestFS().featureSelection(ds)
#ds = PercentilFS().featureSelection(ds)
#ds = RFECVFS().featureSelection(ds)
#ds = SelectFromModelFS().featureSelection(ds)

ds.get_shape()
print(len(np.where(ds.y==0)[0]), len(np.where(ds.y==1)[0]))

pca = PCA().runUnsupervised(ds)


splitter = SingletaskStratifiedSplitter()

train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=ds, frac_train=0.6, frac_valid=0.2, frac_test=0.2)


print(len(np.where(train_dataset.y==0)[0]), len(np.where(train_dataset.y==1)[0]))

#train_dataset = RandomOverSampler().sample(train_dataset)
train_dataset = SMOTEENN().sample(train_dataset)


print(len(np.where(train_dataset.y==0)[0]), len(np.where(train_dataset.y==1)[0]))


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

rf = RandomForestClassifier()
svm = SVC()

model = SklearnModel(model=rf)

#print(model.cross_validate(ds, Metric(roc_auc_score)))

print("#############################")
# model training
print('Fiting Model: ')
model.fit(train_dataset)
print("#############################")

valid_preds = model.predict(valid_dataset)
test_preds = model.predict(test_dataset)


metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
           Metric(classification_report)]
print("#############################")
# evaluate the model
print('Training Dataset: ')
train_score = model.evaluate(train_dataset, metrics)
print("#############################")
print('Validation Dataset: ')
valid_score = model.evaluate(valid_dataset, metrics)
print("#############################")
print('Test Dataset: ')
test_score = model.evaluate(test_dataset, metrics)
print("#############################")

def rf_model_builder(n_estimators, max_features, class_weight, model_dir=None):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, class_weight=class_weight)
    return SklearnModel(rf_model, model_dir)

params_dict_rf = {"n_estimators": [10, 100],
                  "max_features": ["auto", "sqrt", "log2", None],
                  "class_weight": [{0: 1., 1: 1.}]}#, {0: 1., 1: 5}, {0: 1., 1: 10}]
                  #}

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
best_rf.evaluate(test_dataset, metrics)

