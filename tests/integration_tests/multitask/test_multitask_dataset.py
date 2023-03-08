import os
from abc import ABC
from copy import copy
from unittest import TestCase

import numpy as np
from deepchem.models import GraphConvModel
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

from deepmol.compound_featurization import MorganFingerprint, ConvMolFeat
from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.models import DeepChemModel, KerasModel
from deepmol.models.base_models import create_dense_model, basic_multitask_dnn
from deepmol.splitters import RandomSplitter
from tests import TEST_DIR


class MultitaskBaseTestCase(ABC):

    def setUp(self):
        data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(data_path, "tox21.csv")
        loader = CSVLoader(dataset,
                           smiles_field='smiles',
                           id_field='mol_id',
                           labels_fields=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                                          'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'])
        self.multitask_dataset = loader.create_dataset(sep=",")

    def tearDown(self):
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')


class TestMultitaskDataset(MultitaskBaseTestCase, TestCase):

    def test_multitask_deepchem_model(self):
        md = copy(self.multitask_dataset)
        # replace np.nan in y by 0
        md._y = np.nan_to_num(md.y)
        ConvMolFeat().featurize(md)
        train, test = RandomSplitter().train_test_split(md, frac_train=0.8, seed=123)
        graph = GraphConvModel(n_tasks=md.n_tasks)
        model_graph = DeepChemModel(graph)
        model_graph.fit(train)
        test_preds = model_graph.predict(test)
        self.assertEqual(len(test_preds), len(test))
        for pred in test_preds:
            for p in pred:
                self.assertAlmostEqual(sum(p), 1, delta=0.0001)

        metrics = [Metric(roc_auc_score)]

        evaluate = model_graph.evaluate(md, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], None)
        self.assertTrue('roc_auc_score' in evaluate[0].keys())

    def test_multitask_keras_model(self):
        md = copy(self.multitask_dataset)
        # keep only the first 3 tasks
        md._y = md.y[:, :3]
        md.label_names = md.label_names[:3]
        # replace np.nan in y by 0
        md._y = np.nan_to_num(md.y)
        MorganFingerprint(radius=2, size=1024).featurize(md)
        train, test = RandomSplitter().train_test_split(md, frac_train=0.8, seed=123)
        mt_model = basic_multitask_dnn(input_shape=(1024,),
                                       task_names=md.label_names,
                                       losses=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                                       metrics=['accuracy', 'accuracy', 'accuracy'])
        model = KerasModel(mt_model, epochs=2, mode='multitask')
        model.fit(train)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
        a, b = model.evaluate(test, metrics=metrics)
        self.assertIn('roc_auc_score', a.keys())
        self.assertIn('precision_score', a.keys())
        self.assertIn('accuracy_score', a.keys())
        self.assertIsNone(b)
        c, d = model.evaluate(test, metrics=metrics, per_task_metrics=True)
        self.assertIn('roc_auc_score', c.keys())
        self.assertIn('precision_score', c.keys())
        self.assertIn('accuracy_score', c.keys())
        self.assertEqual(len(d['roc_auc_score']), 3)
        self.assertEqual(len(d['precision_score']), 3)
        self.assertEqual(len(d['accuracy_score']), 3)
