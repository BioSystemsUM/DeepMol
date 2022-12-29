from unittest import TestCase

from deepchem.models import GraphConvModel
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report

from deepmol.compound_featurization import ConvMolFeat
from deepmol.metrics import Metric
from deepmol.models import DeepChemModel
from deepmol.parameter_optimization import HyperparameterOptimizerCV
from deepmol.splitters import SingletaskStratifiedSplitter
from unit_tests.models.test_models import ModelsTestCase


class TestDeepChemHyperparameterOptimization(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        ds = ConvMolFeat().featurize(self.mini_dataset_to_test)
        splitter = SingletaskStratifiedSplitter()
        train_dataset, test_dataset = splitter.train_test_split(ds)

        def graphconv_builder(graph_conv_layers, batch_size=256, epochs=5):
            graph = GraphConvModel(n_tasks=1, graph_conv_layers=graph_conv_layers, batch_size=batch_size,
                                   mode='classification')
            return DeepChemModel(graph, model_dir=None, epochs=epochs)

        model_graph = HyperparameterOptimizerCV(model_builder=graphconv_builder)

        best_model, _, _ = model_graph.hyperparameter_search(train_dataset=train_dataset, metric="roc_auc",
                                                             n_iter_search=2,
                                                             cv=2, params_dict={'graph_conv_layers':
                                                                                    [[64, 64], [32, 32]]},
                                                             model_type="deepchem")

        test_preds = best_model.predict(test_dataset)
        self.assertEqual(len(test_preds), len(test_dataset))
        for pred in test_preds:
            self.assertAlmostEqual(sum(pred), 1, delta=0.0001)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        evaluate = best_model.evaluate(ds, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], None)
        self.assertTrue('roc_auc_score' in evaluate[0].keys())
        self.assertTrue('precision_score' in evaluate[0].keys())
        self.assertTrue('accuracy_score' in evaluate[0].keys())
        self.assertTrue('confusion_matrix' in evaluate[0].keys())
        self.assertTrue('classification_report' in evaluate[0].keys())
