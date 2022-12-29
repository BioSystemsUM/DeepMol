from unittest import TestCase

from deepchem.models import GraphConvModel

from deepmol.compound_featurization import ConvMolFeat
from deepmol.metrics.metrics_functions import roc_auc_score, precision_score, accuracy_score, confusion_matrix, \
    classification_report

from deepmol.metrics import Metric
from deepmol.models import DeepChemModel
from deepmol.splitters import SingletaskStratifiedSplitter
from unit_tests.models.test_models import ModelsTestCase


class TestDeepChemModel(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        ds = ConvMolFeat().featurize(self.mini_dataset_to_test)
        splitter = SingletaskStratifiedSplitter()
        train_dataset, test_dataset = splitter.train_test_split(ds)

        graph = GraphConvModel(n_tasks=1, mode='classification')
        model_graph = DeepChemModel(graph)

        model_graph.fit(train_dataset)
        test_preds = model_graph.predict(test_dataset)
        self.assertEqual(len(test_preds), len(test_dataset))
        for pred in test_preds:
            self.assertAlmostEqual(sum(pred), 1, delta=0.0001)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        evaluate = model_graph.evaluate(ds, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], None)
        self.assertTrue('roc_auc_score' in evaluate[0].keys())
        self.assertTrue('precision_score' in evaluate[0].keys())
        self.assertTrue('accuracy_score' in evaluate[0].keys())
        self.assertTrue('confusion_matrix' in evaluate[0].keys())
        self.assertTrue('classification_report' in evaluate[0].keys())
