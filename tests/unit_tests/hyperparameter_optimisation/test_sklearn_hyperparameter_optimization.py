from typing import Callable
from unittest import TestCase

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC

from deepmol.metrics import Metric
from deepmol.parameter_optimization import HyperparameterOptimizerCV, HyperparameterOptimizerValidation
from deepmol.parameter_optimization._utils import validate_metrics, _convert_hyperparam_dict_to_filename
from unit_tests.models.test_models import ModelsTestCase


class TestSklearnHyperparameterOptimization(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        train_dataset, test_dataset = self.binary_dataset, self.binary_dataset_test

        optimizer = HyperparameterOptimizerValidation(SVC)
        params_dict_svc = {"C": [1.0, 1.2, 0.8]}
        best_svm, best_hyperparams, all_results = optimizer.hyperparameter_search(train_dataset=train_dataset,
                                                                                  valid_dataset=test_dataset,
                                                                                  metric=Metric(accuracy_score),
                                                                                  maximize_metric=True,
                                                                                  n_iter_search=2,
                                                                                  params_dict=params_dict_svc,
                                                                                  model_type="sklearn")

        # Evaluate model
        result = best_svm.evaluate(test_dataset, [Metric(accuracy_score)])

        best_model_name = _convert_hyperparam_dict_to_filename(best_hyperparams)
        self.assertEqual(all_results[best_model_name], result[0]["accuracy_score"])

    def test_hyperparameter_optimization_cv(self):
        train_dataset, test_dataset = self.binary_dataset, self.binary_dataset_test

        def rf_model_builder(n_estimators=10, class_weight=None):
            if class_weight is None:
                class_weight = {0: 1., 1: 1.}
            rf_model = RandomForestClassifier(n_estimators=n_estimators,
                                              class_weight=class_weight)
            return rf_model

        params_dict_rf = {"n_estimators": [10, 100],
                          "class_weight": [{0: 1., 1: 1.}, {0: 1., 1: 5}, {0: 1., 1: 10}]
                          }

        optimizer = HyperparameterOptimizerCV(rf_model_builder)

        metric = Metric(roc_auc_score)

        best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(train_dataset=train_dataset,
                                                                                 metric=metric,
                                                                                 n_iter_search=2,
                                                                                 cv=2, params_dict=params_dict_rf,
                                                                                 model_type="sklearn")

        self.assertEqual(len(all_results['mean_test_score']), 2)
        self.assertEqual(all_results['params'][np.argmax(all_results['mean_test_score'])], best_hyperparams)

        best_rf, best_hyperparams, all_results = optimizer.hyperparameter_search(train_dataset=train_dataset,
                                                                                 metric=Metric(roc_auc_score),
                                                                                 n_iter_search=2,
                                                                                 cv=2, params_dict=params_dict_rf,
                                                                                 model_type="sklearn")

        self.assertEqual(len(all_results['mean_test_score']), 2)
        self.assertEqual(all_results['params'][np.argmax(all_results['mean_test_score'])], best_hyperparams)

        with self.assertRaises(AttributeError):
            optimizer.hyperparameter_search(train_dataset=train_dataset,
                                            metric="not_a_metric",
                                            n_iter_search=2,
                                            cv=2, params_dict=params_dict_rf,
                                            model_type="sklearn")

    def test_validate_metrics(self):
        metric = validate_metrics(roc_auc_score)
        self.assertIsInstance(metric, Callable)

        with self.assertRaises(ValueError):
            validate_metrics("not_a_metric")

        metric = validate_metrics(Metric(roc_auc_score))
        self.assertIsInstance(metric, Callable)

        metric = validate_metrics("roc_auc")
        self.assertIsInstance(metric, str)

    def test_aucs(self):
        train_dataset, test_dataset = self.binary_dataset, self.binary_dataset_test

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
                                                                                 maximize_metric=True,
                                                                                 n_iter_search=2,
                                                                                 params_dict=params_dict_rf,
                                                                                 model_type="sklearn")

        best_model_name = _convert_hyperparam_dict_to_filename(best_hyperparams)

        # Evaluate model
        result = best_rf.evaluate(test_dataset, metrics)

        self.assertEqual(all_results[best_model_name], result[0]["roc_auc_score"])
