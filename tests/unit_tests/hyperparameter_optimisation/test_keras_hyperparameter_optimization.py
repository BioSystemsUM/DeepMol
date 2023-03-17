from typing import Callable
from unittest import TestCase

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.svm import SVC
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

from deepmol.metrics import Metric
from deepmol.parameter_optimization import HyperparameterOptimizerCV, HyperparameterOptimizerValidation
from deepmol.parameter_optimization._utils import validate_metrics, _convert_hyperparam_dict_to_filename
from unit_tests.models.test_models import ModelsTestCase


class TestSklearnHyperparameterOptimization(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        train_dataset, test_dataset = self.binary_dataset, self.binary_dataset_test

        def create_model(input_dim, optimizer='adam', dropout=0.5):
            # create model
            model = Sequential()
            model.add(Dense(12, input_dim=input_dim, activation='relu'))
            model.add(Dropout(dropout))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            return model

        optimizer = HyperparameterOptimizerValidation(create_model)
        params_dict_dense = {
            "input_dim": [train_dataset.X.shape[1]],
            "dropout": [0.5, 0.6, 0.7],
            "optimizer": ["adam", "rmsprop"]
        }
        best_svm, best_hyperparams, all_results = optimizer.hyperparameter_search(train_dataset=train_dataset,
                                                                                  valid_dataset=test_dataset,
                                                                                  metric=Metric(accuracy_score),
                                                                                  maximize_metric=True,
                                                                                  n_iter_search=2,
                                                                                  params_dict=params_dict_dense)

        # Evaluate model
        result = best_svm.evaluate(test_dataset, [Metric(accuracy_score)])

        best_model_name = _convert_hyperparam_dict_to_filename(best_hyperparams)
        self.assertEqual(all_results[best_model_name], result[0]["accuracy_score"])
