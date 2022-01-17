import os
from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report

from compoundFeaturization.rdkitFingerprints import MorganFingerprint
from loaders.Loaders import CSVLoader, SDFLoader
from metrics.Metrics import Metric
from models.ensembles import VotingClassifier
from models.sklearnModels import SklearnModel
from splitters.splitters import SingletaskStratifiedSplitter


class TestEnsembles(TestCase):

    def setUp(self) -> None:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(".")))
        dataset = os.path.join(dir_path, "tests", "data", "dataset_sweet_3d_balanced.sdf")
        loader = SDFLoader(dataset,
                           labels_fields='_SWEET')

        self.binary_dataset = loader.create_dataset()

        dataset = loader.create_dataset()
        splitter = SingletaskStratifiedSplitter()
        MorganFingerprint().featurize(dataset)
        self.train_dataset, self.test_dataset = splitter.train_test_split(dataset)

    def testVotingClassifier(self):
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


