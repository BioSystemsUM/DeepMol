import os
from abc import ABC
from copy import copy
from random import random
from unittest import TestCase, skip

import keras
import numpy as np
import pandas as pd
from deepchem.models import GraphConvModel
from keras import layers
from sklearn.metrics import precision_score, recall_score, r2_score, f1_score

from deepmol.compound_featurization import MorganFingerprint, ConvMolFeat
from deepmol.loaders import CSVLoader
from deepmol.loggers import Logger
from deepmol.metrics import Metric
from deepmol.models import DeepChemModel, KerasModel
from deepmol.models.base_models import basic_multitask_dnn
from deepmol.models.keras_model_builders import keras_1d_cnn_model_builder
from deepmol.splitters import RandomSplitter
from tests import TEST_DIR

import tensorflow as tf

import shutil


class MultitaskBaseTestCase(ABC):

    def setUp(self):
        data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(data_path, "tox21_small_.csv")

        loader = CSVLoader(dataset,
                           smiles_field='smiles',
                           id_field='mol_id',
                           labels_fields=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                                          'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'])
        self.multitask_dataset = loader.create_dataset(sep=",")

        dataset_multilabel = os.path.join(data_path, "multilabel_classification_dataset.csv")
        loader = CSVLoader(dataset_multilabel,
                           smiles_field='smiles',
                           labels_fields=['C00341', 'C01789', 'C00078'],
                           mode=['classification', 'classification', 'classification'],
                           shard_size=100)
        self.dataset_multilabel = loader.create_dataset(sep=",")

        self.dataset_multilabel_regression = copy(self.dataset_multilabel)
        # change dataset.y to regression (3 labels with random float values between 0 and 10)
        self.dataset_multilabel_regression._y = np.array([[random() * 10 for _ in range(3)] for _ in
                                                          range(len(self.dataset_multilabel_regression))])
        self.dataset_multilabel_regression.mode = ['regression', 'regression', 'regression']

        self.dataset_multilabel_regression = copy(self.dataset_multilabel)
        # change dataset.y to regression (3 labels with random float values between 0 and 10)
        self.dataset_multilabel_regression._y = np.array([[random() * 10 for _ in range(3)] for _ in
                                                          range(len(self.dataset_multilabel_regression))])
        self.dataset_multilabel_regression.mode = ['regression', 'regression', 'regression']

        tf.config.set_visible_devices([], 'GPU')

    def tearDown(self):
        # Close logger file handlers to release the file
        singleton_instance = Logger()
        singleton_instance.close_handlers()

        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

        if os.path.exists('test_multitask_keras_model'):
            # remove directory recursively

            shutil.rmtree('test_multitask_keras_model')


def create_multitask_regression_model(input_shape, tasks, hidden_units=64):
    # Input layer
    input_layer = layers.Input(shape=input_shape, name='input_data')

    # Hidden layer
    hidden_layer = layers.Dense(hidden_units, activation='relu')(input_layer)

    # Output layers for each regression task
    output_layers = []
    for i in range(len(tasks)):
        output_layer = layers.Dense(1, name=tasks[i], activation="linear")(hidden_layer)
        output_layers.append(output_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layers)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model


class TestMultitaskDataset(MultitaskBaseTestCase, TestCase):

    def test_multitask_deepchem_model(self):
        md = copy(self.multitask_dataset)
        # replace np.nan in y by 0
        md._y = np.nan_to_num(md.y)
        ConvMolFeat().featurize(md, inplace=True)
        train, test = RandomSplitter().train_test_split(md, frac_train=0.8, seed=123)
        graph = GraphConvModel
        model_graph = DeepChemModel(graph, n_tasks=md.n_tasks)
        model_graph.fit(train)
        test_preds = model_graph.predict(test)
        self.assertEqual(len(test_preds), len(test))

        metrics = [Metric(precision_score, average="macro")]

        evaluate = model_graph.evaluate(test, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], {})
        self.assertTrue('precision_score' in evaluate[0].keys())
        predictions = model_graph.predict(test)
        precision_score_value = precision_score(test.y, predictions, average="macro")
        self.assertEqual(evaluate[0]['precision_score'], precision_score_value)

    # @skip("Skip, it takes too long for CI")
    def test_multitask_model_with_baseline_cnn(self):
        md = copy(self.multitask_dataset)
        # keep only the first 3 tasks
        # replace np.nan in y by 0
        md._y = np.nan_to_num(md.y)
        md.mode = ["classification"] * len(md.label_names)
        MorganFingerprint(radius=2, size=1024).featurize(md, inplace=True)
        train, test = RandomSplitter().train_test_split(md, frac_train=0.8, seed=123)
        mt_model = keras_1d_cnn_model_builder(input_dim=1024,
                                              n_tasks=len(md.label_names),
                                              label_names=list(md.label_names),
                                              last_layers_activations=['sigmoid'] * len(md.label_names),
                                              losses=['binary_crossentropy'] * len(md.label_names),
                                              last_layers_units=[1] * len(md.label_names))

        model = KerasModel(mt_model, epochs=2, mode=["classification"] * len(md.label_names))
        model.fit(train)

        metrics = [Metric(precision_score, average="macro"), Metric(recall_score, average="macro"),
                   Metric(f1_score, average="macro")]

        a, b = model.evaluate(test, metrics=metrics)
        self.assertIn('precision_score', a.keys())
        self.assertIn('recall_score', a.keys())
        self.assertEqual(b, {})
        c, d = model.evaluate(test, metrics=metrics, per_task_metrics=True)
        self.assertIn('precision_score', c.keys())
        self.assertIn('recall_score', c.keys())
        f1_first = c["f1_score"]

        predictions = model.predict(test)
        precision_score_value = precision_score(test.y, predictions, average="macro")
        self.assertEqual(a['precision_score'], precision_score_value)

        predictions1 = model.predict(test)
        model.save('test_multitask_keras_model')

        model_loaded = KerasModel.load('test_multitask_keras_model')
        predictions2 = model_loaded.predict(test)
        assert np.array_equal(predictions1, predictions2)

        f1_second = f1_score(test.y, predictions2, average="macro")
        self.assertEqual(f1_first, f1_second)

    def test_multitask_keras_model(self):
        md = copy(self.multitask_dataset)
        # keep only the first 3 tasks
        md._y = md.y[:, :3]
        md.label_names = md.label_names[:3]
        # replace np.nan in y by 0
        md._y = np.nan_to_num(md.y)
        md.mode = ["classification", "classification", "classification"]
        MorganFingerprint(radius=2, size=1024).featurize(md, inplace=True)
        train, test = RandomSplitter().train_test_split(md, frac_train=0.8, seed=123)
        mt_model = basic_multitask_dnn(input_shape=(1024,),
                                       task_names=md.label_names,
                                       losses=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                                       metrics=['accuracy', 'accuracy', 'accuracy'])
        model = KerasModel(mt_model, epochs=150, mode=["classification", "classification", "classification"])
        model.fit(train)

        metrics = [Metric(precision_score, average="macro"), Metric(recall_score, average="macro")]
        a, b = model.evaluate(test, metrics=metrics)
        self.assertIn('precision_score', a.keys())
        self.assertIn('recall_score', a.keys())
        self.assertEqual(b, {})
        c, d = model.evaluate(test, metrics=metrics, per_task_metrics=True)
        self.assertIn('precision_score', c.keys())
        self.assertIn('recall_score', c.keys())
        self.assertEqual(len(d['precision_score']), 3)
        self.assertEqual(len(d['recall_score']), 3)

        predictions = model.predict(test)
        precision_score_value = precision_score(test.y, predictions, average="macro")
        self.assertEqual(a['precision_score'], precision_score_value)

        predictions1 = model.predict(test)
        model.save('test_multitask_keras_model')

        model_loaded = KerasModel.load('test_multitask_keras_model')
        predictions2 = model_loaded.predict(test)
        assert np.array_equal(predictions1, predictions2)

    def test_multitask_keras_model_regression(self):
        MorganFingerprint(radius=2, size=1024).featurize(self.dataset_multilabel_regression, inplace=True)
        train, test = RandomSplitter().train_test_split(self.dataset_multilabel_regression, seed=124)

        model = KerasModel(create_multitask_regression_model(input_shape=(1024,),
                                                             tasks=list(
                                                                 self.dataset_multilabel_regression.label_names)),
                           epochs=2,
                           mode=["regression", "regression", "regression"])
        model.fit(train)

        metrics = [Metric(r2_score)]
        a, b = model.evaluate(test, metrics=metrics)
        self.assertIn('r2_score', a.keys())
        self.assertEqual(b, {})
        c, d = model.evaluate(test, metrics=metrics, per_task_metrics=True)
        self.assertIn('r2_score', c.keys())
        self.assertEqual(len(d['r2_score']), 3)

        predictions = model.predict(test)
        r2_score_value = r2_score(test.y, predictions)
        self.assertEqual(a['r2_score'], r2_score_value)

        predictions1 = model.predict(test)
        model.save('test_multitask_keras_model')

        model_loaded = KerasModel.load('test_multitask_keras_model')
        predictions2 = model_loaded.predict(test)
        assert np.array_equal(predictions1, predictions2)

        predictions, _ = model.evaluate(test, metrics=metrics, per_task_metrics=True)
        self.assertEqual(predictions['r2_score'], c['r2_score'])
