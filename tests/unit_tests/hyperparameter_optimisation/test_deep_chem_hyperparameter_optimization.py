import os
import pickle
from unittest import TestCase
from unittest.mock import MagicMock

from deepchem.feat import ConvMolFeaturizer
from deepchem.models import GraphConvModel
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report

from deepmol.datasets import SmilesDataset
from deepmol.metrics import Metric
from deepmol.models import DeepChemModel
from deepmol.parameter_optimization import HyperparameterOptimizerCV
from unit_tests.models.test_models import ModelsTestCase

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestDeepChemHyperparameterOptimization(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        ds_train = self.binary_dataset
        ds_train.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 100)
        ds_train.select_to_split.side_effect = lambda arg: MagicMock(spec=SmilesDataset,
                                                                     X=ds_train.X[arg],
                                                                     y=ds_train.y[arg],
                                                                     n_tasks=1,
                                                                     label_names=['binary_label'],
                                                                     mode='classification',
                                                                     ids=ds_train.ids[arg])
        ds_test = self.binary_dataset_test
        ds_test.X = ConvMolFeaturizer().featurize([MolFromSmiles('CCC')] * 10)

        def graphconv_builder(graph_conv_layers, batch_size=256, epochs=5):
            graph = GraphConvModel(n_tasks=1, graph_conv_layers=graph_conv_layers, batch_size=batch_size,
                                   mode='classification')
            return DeepChemModel(graph, model_dir=None, epochs=epochs)

        model_graph = HyperparameterOptimizerCV(model_builder=graphconv_builder)

        best_model, _, _ = model_graph.hyperparameter_search(train_dataset=ds_train, metric=Metric(roc_auc_score),
                                                             n_iter_search=2,
                                                             cv=2, params_dict={'graph_conv_layers': [[64, 64],
                                                                                                      [32, 32]]},
                                                             model_type="deepchem")

        test_preds = best_model.predict(ds_test)
        self.assertEqual(len(test_preds), len(ds_test))
        for pred in test_preds:
            self.assertAlmostEqual(sum(pred), 1, delta=0.0001)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        evaluate = best_model.evaluate(ds_test, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], {})
        self.assertTrue('roc_auc_score' in evaluate[0].keys())
        self.assertTrue('precision_score' in evaluate[0].keys())
        self.assertTrue('accuracy_score' in evaluate[0].keys())
        self.assertTrue('confusion_matrix' in evaluate[0].keys())
        self.assertTrue('classification_report' in evaluate[0].keys())
