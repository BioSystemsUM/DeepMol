import os
from unittest import TestCase

import numpy as np
from deepchem.feat import ConvMolFeaturizer
from deepchem.models import GraphConvModel
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import f1_score

from deepmol.metrics.metrics_functions import roc_auc_score, precision_score, accuracy_score, confusion_matrix

from deepmol.metrics import Metric
from deepmol.models import DeepChemModel
from unit_tests.models.test_models import ModelsTestCase

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestDeepChemModel(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        ds_train = self.binary_dataset
        ds_train.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 100)
        ds_test = self.binary_dataset_test
        ds_test.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 10)

        graph = GraphConvModel(n_tasks=1, mode='classification')
        model_graph = DeepChemModel(graph)

        model_graph.fit(ds_train)
        test_preds = model_graph.predict(ds_test)
        self.assertEqual(len(test_preds), len(ds_test))
        for pred in test_preds:
            self.assertAlmostEqual(sum(pred), 1, delta=0.0001)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix)]

        evaluate = model_graph.evaluate(ds_test, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], {})

        roc_value = roc_auc_score(ds_test.y, test_preds[:, 1])
        self.assertEqual(evaluate[0]['roc_auc_score'], roc_value)
        precision_value = precision_score(ds_test.y, np.round(test_preds[:, 1]))
        self.assertEqual(evaluate[0]['precision_score'], precision_value)
        accuracy_value = accuracy_score(ds_test.y, np.round(test_preds[:, 1]))
        self.assertEqual(evaluate[0]['accuracy_score'], accuracy_value)
        confusion_matrix_value = confusion_matrix(ds_test.y, np.round(test_preds[:, 1]))
        self.assertEqual(evaluate[0]['confusion_matrix'][0][0], confusion_matrix_value[0][0])
        self.assertEqual(evaluate[0]['confusion_matrix'][0][1], confusion_matrix_value[0][1])
        self.assertEqual(evaluate[0]['confusion_matrix'][1][0], confusion_matrix_value[1][0])
        self.assertEqual(evaluate[0]['confusion_matrix'][1][1], confusion_matrix_value[1][1])

    def test_multiclass(self):
        ds_train = self.multitask_dataset
        ds_train.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 100)
        ds_test = self.multitask_dataset_test
        ds_test.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 10)

        graph = GraphConvModel(n_tasks=ds_train.n_tasks, mode='classification')
        model_graph = DeepChemModel(graph)

        model_graph.fit(ds_train)
        test_preds = model_graph.predict(ds_test)
        self.assertEqual(len(test_preds), len(ds_test))
        y_preds = []
        for pred in test_preds:
            y_p = []
            for p in pred:
                self.assertAlmostEqual(sum(p), 1, delta=0.0001)
                y_p.append(p[1])
            y_preds.append(y_p)

        metrics = [Metric(f1_score, average='micro'), Metric(precision_score, average='micro')]

        evaluate = model_graph.evaluate(ds_test, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], {})

        precision_score_value = precision_score(self.multitask_dataset_test.y, np.round(np.array(y_preds)),
                                                average='micro')
        f1_score_value = f1_score(self.multitask_dataset_test.y, np.round(np.array(y_preds)), average='micro')
        self.assertEqual(precision_score_value, evaluate[0]["precision_score"])
        self.assertEqual(f1_score_value, evaluate[0]["f1_score"])
