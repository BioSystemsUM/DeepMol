import os
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from deepchem.feat import ConvMolFeaturizer, MolGraphConvFeaturizer
from deepchem.models import GraphConvModel, TextCNNModel, GCNModel, MultitaskClassifier
from deepchem.models.layers import DTNNEmbedding, Highway
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import f1_score

from deepmol.datasets import SmilesDataset
from deepmol.metrics.metrics_functions import roc_auc_score, precision_score, accuracy_score, confusion_matrix

from deepmol.metrics import Metric
from deepmol.models import DeepChemModel
from deepmol.splitters import RandomSplitter
from deepmol.compound_featurization import MorganFingerprint
from unit_tests.models.test_models import ModelsTestCase

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestDeepChemModel(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        ds_train = self.binary_dataset
        ds_train.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 100)
        ds_test = self.binary_dataset_test
        ds_test.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 10)

        # graph = GraphConvModel(n_tasks=1, mode='classification')
        model_graph = DeepChemModel(GraphConvModel, n_tasks=1, mode='classification')

        model_graph.fit(ds_train)
        test_preds = model_graph.predict(ds_test)
        self.assertEqual(len(test_preds), len(ds_test))

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix)]

        evaluate = model_graph.evaluate(ds_test, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], {})

        roc_value = roc_auc_score(ds_test.y, test_preds)
        self.assertEqual(evaluate[0]['roc_auc_score'], roc_value)
        precision_value = precision_score(ds_test.y, np.round(test_preds))
        self.assertEqual(evaluate[0]['precision_score'], precision_value)
        accuracy_value = accuracy_score(ds_test.y, np.round(test_preds))
        self.assertEqual(evaluate[0]['accuracy_score'], accuracy_value)
        confusion_matrix_value = confusion_matrix(ds_test.y, np.round(test_preds))
        self.assertEqual(evaluate[0]['confusion_matrix'][0][0], confusion_matrix_value[0][0])
        self.assertEqual(evaluate[0]['confusion_matrix'][0][1], confusion_matrix_value[0][1])
        self.assertEqual(evaluate[0]['confusion_matrix'][1][0], confusion_matrix_value[1][0])
        self.assertEqual(evaluate[0]['confusion_matrix'][1][1], confusion_matrix_value[1][1])

    def test_multiclass(self):
        ds_train = self.multitask_dataset
        ds_train.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 100)
        ds_test = self.multitask_dataset_test
        ds_test.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 10)

        model_graph = DeepChemModel(GraphConvModel, n_tasks=ds_train.n_tasks, mode='classification')

        model_graph.fit(ds_train)
        test_preds = model_graph.predict(ds_test)
        self.assertEqual(len(test_preds), len(ds_test))

        metrics = [Metric(f1_score, average='micro'), Metric(precision_score, average='micro')]

        evaluate = model_graph.evaluate(ds_test, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], {})

        precision_score_value = precision_score(self.multitask_dataset_test.y, np.round(np.array(test_preds)),
                                                average='micro')
        f1_score_value = f1_score(self.multitask_dataset_test.y, np.round(np.array(test_preds)), average='micro')
        self.assertEqual(precision_score_value, evaluate[0]["precision_score"])
        self.assertEqual(f1_score_value, evaluate[0]["f1_score"])

    def test_save(self):
        ds_train = self.multitask_dataset
        ds_train.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 10)
        ds_test = self.multitask_dataset_test
        ds_test.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 10)

        model_graph = DeepChemModel(GraphConvModel, n_tasks=ds_train.n_tasks, mode='classification', epochs=3)

        model_graph.fit(ds_train)
        test_preds = model_graph.predict(ds_test)

        model_graph.save("test_model")
        model_graph_loaded = DeepChemModel.load("test_model")
        self.assertEqual(model_graph.n_tasks, ds_train.n_tasks)
        self.assertEqual(model_graph.epochs, 3)
        new_predictions = model_graph_loaded.predict(ds_test)
        self.assertTrue(np.array_equal(test_preds, new_predictions))

    def test_cross_validate(self):
        ds_train = self.binary_dataset
        ds_train.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 100)
        ds_train.select_to_split.side_effect = lambda arg: MagicMock(spec=SmilesDataset,
                                                                     X=ds_train.X[arg],
                                                                     y=ds_train.y[arg],
                                                                     n_tasks=1,
                                                                     label_names=['binary_label'],
                                                                     mode='classification',
                                                                     ids=ds_train.ids[arg])

        model = DeepChemModel(GraphConvModel, n_tasks=1, mode='classification', epochs=3)

        best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, avg_train_score, \
            avg_test_score = model.cross_validate(ds_train, metric=Metric(roc_auc_score), folds=3)
        self.assertIsNotNone(best_model)
        self.assertIsInstance(train_score_best_model, float)
        self.assertIsInstance(test_score_best_model, float)
        self.assertEqual(len(train_scores), 3)
        self.assertEqual(len(test_scores), 3)
        self.assertIsInstance(avg_train_score, float)
        self.assertIsInstance(avg_test_score, float)

        splitter = RandomSplitter()
        best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, avg_train_score, \
            avg_test_score = model.cross_validate(self.binary_dataset,
                                                  metric=Metric(roc_auc_score),
                                                  splitter=splitter,
                                                  folds=3)
        self.assertIsNotNone(best_model)
        self.assertIsInstance(train_score_best_model, float)
        self.assertIsInstance(test_score_best_model, float)
        self.assertEqual(len(train_scores), 3)
        self.assertEqual(len(test_scores), 3)
        self.assertIsInstance(avg_train_score, float)
        self.assertIsInstance(avg_test_score, float)

    def test_save_text_cnn(self):
        self.binary_dataset.ids = self.binary_dataset.smiles
        self.binary_dataset.mode = 'classification'
        self.binary_dataset.X = self.binary_dataset.smiles

        char_dict, length = TextCNNModel.build_char_dict(self.binary_dataset)

        model = DeepChemModel(TextCNNModel, n_tasks=1, char_dict=char_dict, seq_length=length, epochs=10)
        model.fit(self.binary_dataset)
        test_predict = model.predict(self.binary_dataset)
        model.save("test_model")

        new_model_loaded = DeepChemModel.load("test_model")
        new_predict = new_model_loaded.predict(self.binary_dataset)
        self.assertTrue(np.array_equal(test_predict, new_predict))

    def test_torch_model_save(self):
        self.binary_dataset.X = MolGraphConvFeaturizer().featurize([MolFromSmiles('CCC')] * 100)

        model = DeepChemModel(GCNModel, n_tasks=1, graph_conv_layers=[32, 32], activation=None,
                       residual=True, batchnorm=False, predictor_hidden_feats=64,
                       dropout=0.25, predictor_dropout=0.25,
                       learning_rate=1e-3,
                       batch_size=20, mode="classification",
                       device="cpu", epochs=10)
        
        model.fit(self.binary_dataset)
        test_predict = model.predict(self.binary_dataset)
        metrics = [Metric(f1_score, average='micro'), Metric(precision_score, average='micro')]

        evaluation = model.evaluate(self.binary_dataset, metrics)
        model.save("test_model")
        new_model = DeepChemModel.load("test_model")
        new_predict = new_model.predict(self.binary_dataset)
        new_evaluation = new_model.evaluate(self.binary_dataset, metrics)

        self.assertTrue(np.array_equal(test_predict, new_predict))

        self.assertEqual(evaluation, new_evaluation)

    def test_save_without_model_dir(self):
        self.binary_dataset.X = MolGraphConvFeaturizer().featurize([MolFromSmiles('CCC')] * 100)

        model = DeepChemModel(GCNModel, n_tasks=1, graph_conv_layers=[32, 32], activation=None,
                       residual=True, batchnorm=False, predictor_hidden_feats=64,
                       dropout=0.25, predictor_dropout=0.25,
                       learning_rate=1e-3,
                       batch_size=20, device="cpu",
                       mode="classification", epochs=10, model_dir="test_model")
        model.fit(self.binary_dataset)
        model.save()
        model = DeepChemModel.load("test_model")
        model.predict(self.binary_dataset)
        self.assertTrue(os.path.exists("test_model"))

    def test_multitask_regressors_and_classifiers(self):
        ds_train = self.multitask_dataset

        model = DeepChemModel(MultitaskClassifier, n_tasks=ds_train.n_tasks, n_features=ds_train.X.shape[1])
        self.assertEqual(model.model.model.mode, "classification")