from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from deepmol.compound_featurization import MorganFingerprint
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.parameter_optimization import HyperparameterOptimizerCV, HyperparameterOptimizerValidation
from deepmol.parameter_optimization.hyperparameter_optimization import _convert_hyperparam_dict_to_filename
from deepmol.splitters import SingletaskStratifiedSplitter
from unit_tests.models.test_models import ModelsTestCase


class TestSklearnHyperparameterOptimization(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        pass

    def test_hyperparameter_optimization_cv(self):

        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)

        MorganFingerprint().featurize(self.larger_dataset_to_test)

        splitter = SingletaskStratifiedSplitter()
        train_dataset, test_dataset = splitter.train_test_split(self.larger_dataset_to_test)

        def rf_model_builder(n_estimators=10, max_features='auto', class_weight=None):
            if class_weight is None:
                class_weight = {0: 1., 1: 1.}
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                              class_weight=class_weight)
            return rf_model

        params_dict_rf = {"n_estimators": [10, 100],
                          "max_features": ["auto", "sqrt", "log2", None],
                          "class_weight": [{0: 1., 1: 1.}, {0: 1., 1: 5}, {0: 1., 1: 10}]
                          }

        optimizer = HyperparameterOptimizerCV(rf_model_builder)

        metrics = [Metric(roc_auc_score)]

        best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(train_dataset=train_dataset,
                                                                                 metric="roc_auc",
                                                                                 n_iter_search=2,
                                                                                 cv=2, params_dict=params_dict_rf,
                                                                                 model_type="sklearn")

        print('#################')
        print(best_hyperparams)
        print(best_rf)
        print(all_results)

        # Evaluate model
        result = best_rf.evaluate(test_dataset, metrics)

    def test_aucs(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)

        MorganFingerprint().featurize(self.larger_dataset_to_test)

        splitter = SingletaskStratifiedSplitter()
        train_dataset, test_dataset = splitter.train_test_split(self.larger_dataset_to_test)

        def rf_model_builder(n_estimators=10, max_features='auto', class_weight=None):
            if class_weight is None:
                class_weight = {0: 1., 1: 1.}
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                              class_weight=class_weight)
            return rf_model

        params_dict_rf = {"n_estimators": [10, 100],
                          "max_features": ["auto", "sqrt", "log2", None],
                          "class_weight": [{0: 1., 1: 1.}, {0: 1., 1: 5}, {0: 1., 1: 10}]
                          }

        optimizer = HyperparameterOptimizerValidation(rf_model_builder)

        metrics = [Metric(roc_auc_score)]

        best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(train_dataset=train_dataset,
                                                                                 valid_dataset=test_dataset,
                                                                                 metric=Metric(roc_auc_score),
                                                                                 n_iter_search=2,
                                                                                 params_dict=params_dict_rf)

        # best_model_name = _convert_hyperparam_dict_to_filename(best_hyperparams)
        #
        # # Evaluate model
        # result = best_rf.evaluate(test_dataset, metrics)
        #
        # self.assertEqual(all_results[best_model_name], result[0]["roc_auc_score"])
