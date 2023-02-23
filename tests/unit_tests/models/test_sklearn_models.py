from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report

from deepmol.compound_featurization import MorganFingerprint
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.splitters import SingletaskStratifiedSplitter
from unit_tests.models.test_models import ModelsTestCase


class TestSklearnModel(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)

        MorganFingerprint().featurize(self.mini_dataset_to_test)

        splitter = SingletaskStratifiedSplitter()
        train_dataset, test_dataset = splitter.train_test_split(self.mini_dataset_to_test, frac_train=0.6)

        model.fit(train_dataset)

        test_preds = model.predict(test_dataset)
        self.assertEqual(len(test_preds), len(test_dataset))
        # evaluate the model
        for pred in test_preds:
            self.assertAlmostEqual(sum(pred), 1, delta=0.0001)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        evaluate = model.evaluate(test_dataset, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], None)
        self.assertTrue('roc_auc_score' in evaluate[0].keys())
        self.assertTrue('precision_score' in evaluate[0].keys())
        self.assertTrue('accuracy_score' in evaluate[0].keys())
        self.assertTrue('confusion_matrix' in evaluate[0].keys())
        self.assertTrue('classification_report' in evaluate[0].keys())

    def test_aucs(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)

        MorganFingerprint().featurize(self.larger_dataset_to_test)

        model.fit(self.larger_dataset_to_test)
        metrics = [Metric(roc_auc_score)]

        # evaluate the model
        print('Training Dataset: ')
        train_score = model.evaluate(self.larger_dataset_to_test, metrics)

        y_pred = model.model.predict_proba(self.larger_dataset_to_test.X)
        y_pred = y_pred[:, 1]
        roc_auc_score_value = roc_auc_score(self.larger_dataset_to_test.y, y_pred)
        self.assertEqual(roc_auc_score_value, train_score[0]["roc_auc_score"])

