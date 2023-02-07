from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from deepmol.compound_featurization import MorganFingerprint
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from unit_tests.models.test_models import ModelsTestCase


class TestDeepChemModel(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)

        MorganFingerprint().featurize(self.mini_dataset_to_test)

        model.fit(self.mini_dataset_to_test)
        metrics = [Metric(roc_auc_score)]

        # evaluate the model
        print('Training Dataset: ')
        train_score = model.evaluate(self.mini_dataset_to_test, metrics)
