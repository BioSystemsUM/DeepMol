import os
from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report

from compound_featurization.rdkit_fingerprints import MorganFingerprint
from loaders.loaders import CSVLoader, SDFLoader
from metrics.metrics import Metric
from models.ensembles import VotingClassifier
from models.sklearn_models import SklearnModel
from splitters.splitters import SingletaskStratifiedSplitter


class TestEnsembles(TestCase):

    def setUp(self) -> None:
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), 'tests', 'data')

        dataset = os.path.join(self.data_path, "dataset_sweet_3d_balanced.sdf")
        loader = SDFLoader(dataset,
                           labels_fields='_SWEET')

        self.binary_dataset = loader.create_dataset()

        dataset = loader.create_dataset()
        splitter = SingletaskStratifiedSplitter()
        MorganFingerprint().featurize(dataset)
        self.train_dataset, self.test_dataset = splitter.train_test_split(dataset)

    def test_soft_voting_classifier(self):
        rf_model = RandomForestClassifier()
        rf_model2 = RandomForestClassifier()

        rf_model = SklearnModel(rf_model)
        rf_model2 = SklearnModel(rf_model2)

        ensemble = VotingClassifier([rf_model, rf_model2])
        ensemble.fit(self.train_dataset)
        predictions = ensemble.predict(self.test_dataset)
        predictions = ensemble.predict(self.test_dataset, proba=True)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        evaluate = ensemble.evaluate(self.test_dataset, metrics)

    def test_hard_voting_classifier(self):
        rf_model = RandomForestClassifier()
        rf_model2 = RandomForestClassifier()

        rf_model = SklearnModel(rf_model)
        rf_model2 = SklearnModel(rf_model2)

        ensemble = VotingClassifier([rf_model, rf_model2], voting="hard")
        ensemble.fit(self.train_dataset)
        predictions = ensemble.predict(self.test_dataset)
        predictions = ensemble.predict(self.test_dataset, proba=True)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        evaluate = ensemble.evaluate(self.test_dataset, metrics)
