import os
from unittest import TestCase

import shap
from shap.maskers import Partition
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from deepmol.compound_featurization import TwoDimensionDescriptors, MorganFingerprint
from deepmol.feature_importance import ShapValues
from deepmol.feature_selection import LowVarianceFS, KbestFS, PercentilFS
from deepmol.loaders import CSVLoader
from deepmol.models import SklearnModel
from deepmol.scalers import MinMaxScaler
from tests import TEST_DIR


class TestShap(TestCase):

    def setUp(self) -> None:
        data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(data_path, "PC-3.csv")
        loader = CSVLoader(dataset,
                           smiles_field='smiles',
                           labels_fields=['pIC50'],
                           shard_size=100)
        self.regression_dataset = loader.create_dataset(sep=",")
        MorganFingerprint().featurize(self.regression_dataset)
        LowVarianceFS(threshold=0.15).select_features(self.regression_dataset)
        self.model = SklearnModel(RandomForestRegressor())
        self.model.fit(self.regression_dataset)

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    def validate_shap(self, explainer, masker, **kwargs):
        shap = ShapValues(explainer=explainer, masker=masker)
        shap.compute_shap(self.regression_dataset, self.model, **kwargs)

    def test_shap(self):
        # self.validate_shap('permutation', None)
        # self.validate_shap('exact', None)
        self.validate_shap('exact', 'partition')
        # self.validate_shap('exact', 'partition')
