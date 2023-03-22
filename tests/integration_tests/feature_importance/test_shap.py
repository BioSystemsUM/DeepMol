import os
from unittest import TestCase

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from deepmol.compound_featurization import TwoDimensionDescriptors
from deepmol.feature_importance import ShapValues
from deepmol.loaders import CSVLoader
from deepmol.models import SklearnModel, KerasModel
from deepmol.models.base_models import basic_dnn_regression

from tests import TEST_DIR


class TestShap(TestCase):

    def setUp(self) -> None:
        data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(data_path, "PC-3.csv")
        loader = CSVLoader(dataset,
                           smiles_field='smiles',
                           labels_fields=['pIC50'],
                           shard_size=5,
                           mode='regression')
        self.regression_dataset = loader.create_dataset(sep=",")
        TwoDimensionDescriptors().featurize(self.regression_dataset)
        print(self.regression_dataset.feature_names)
        # to speed up tests
        self.regression_dataset.select_features_by_name(self.regression_dataset.feature_names[:5])
        self.rf_model = SklearnModel(RandomForestRegressor())
        self.rf_model.fit(self.regression_dataset)
        self.linear_model = SklearnModel(LinearRegression())
        self.linear_model.fit(self.regression_dataset)
        # self.mlp_model = KerasModel(basic_dnn_regression(input_size=5, batch_size=1),
        #                            epochs=2,
        #                            mode='regression',
        #                            loss='mse',
        #                            batch_size=1)
        # self.mlp_model.fit(self.regression_dataset)

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    def validate_shap(self, explainer, masker, **kwargs):
        shap = ShapValues(explainer=explainer, masker=masker)
        shap.compute_shap(self.regression_dataset, self.rf_model, **kwargs)
        self.assertIsNotNone(shap.shap_values)
        #shap.bar_plot()
        #shap.beeswarm_plot()

    def test_shap(self):
        self.validate_shap('explainer', None)
        self.validate_shap('explainer', 'partition')
        self.validate_shap('permutation', None)
        self.validate_shap('permutation', 'partition')
        self.validate_shap('exact', None)
        self.validate_shap('additive', None)
        self.validate_shap('partition', None)
        #self.validate_shap('sampling', None)
        #self.validate_shap('kernel', None)

    def validate_linear_shap(self, explainer, masker, **kwargs):
        shap = ShapValues(explainer=explainer, masker=masker)
        shap.compute_shap(self.regression_dataset, self.linear_model, **kwargs)
        self.assertIsNotNone(shap.shap_values)

    def test_linear_shap(self):
        self.validate_linear_shap('linear', None)

    def validate_dl_shap(self, explainer, masker, **kwargs):
        shap = ShapValues(explainer=explainer, masker=masker)
        shap.compute_shap(self.regression_dataset, self.mlp_model, **kwargs)
        self.assertIsNotNone(shap.shap_values)

    def test_dl_shap(self):
        self.validate_dl_shap('deep', None)
        self.validate_dl_shap('gradient', None)
