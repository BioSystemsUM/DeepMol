import os
from unittest import TestCase

from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from deepmol.compound_featurization import MorganFingerprint
from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.splitters.multitask_splitter import MultiTaskStratifiedSplitter
from tests import TEST_DIR


class TestMultiLabelClassification(TestCase):

    def setUp(self) -> None:
        multilabel_classification_df = os.path.join(TEST_DIR, 'data', "multilabel_classification_dataset.csv")
        loader = CSVLoader(dataset_path=multilabel_classification_df,
                           smiles_field='smiles',
                           id_field='ids',
                           labels_fields=['C00341', 'C01789', 'C00078', 'C00049', 'C00183', 'C03506', 'C00187',
                                          'C00079', 'C00047', 'C01852', 'C00407', 'C00129', 'C00235', 'C00062',
                                          'C00353', 'C00148', 'C00073', 'C00108', 'C00123', 'C00135', 'C00448',
                                          'C00082', 'C00041'],
                           mode='auto')
        # create the dataset
        self.dataset = loader.create_dataset(sep=',', header=0)

    def test_multilabel_classification(self):
        train_dataset, validation_dataset, test_dataset = \
            MultiTaskStratifiedSplitter().train_valid_test_split(self.dataset, frac_train=0.7,
                                                                 frac_valid=0.2,
                                                                 frac_test=0.1)

        dt = DecisionTreeClassifier()
        model = SklearnModel(model=dt)
        MorganFingerprint().featurize(train_dataset, inplace=True)
        MorganFingerprint().featurize(validation_dataset, inplace=True)
        MorganFingerprint().featurize(test_dataset, inplace=True)
        model.fit(train_dataset)

        metrics = [Metric(f1_score)]

        evaluate = model.evaluate(test_dataset, metrics, per_task_metrics=True)
        self.assertEqual(len(evaluate[0]), len(metrics))
        evaluate2 = model.evaluate(validation_dataset, metrics)
        self.assertEqual(len(evaluate2[0]), len(metrics))
