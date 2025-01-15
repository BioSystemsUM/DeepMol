from copy import deepcopy
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from deepmol.compound_featurization.deepchem_featurizers import ConvMolFeat
from deepmol.compound_featurization.rdkit_descriptors import TwoDimensionDescriptors
from deepmol.compound_featurization.rdkit_fingerprints import MorganFingerprint
from deepmol.feature_selection.base_feature_selector import LowVarianceFS
from deepmol.loaders.loaders import CSVLoader
from deepmol.models.deepchem_models import DeepChemModel
from deepmol.models.keras_models import KerasModel
from deepmol.models.sklearn_models import SklearnModel
from deepmol.pipeline.ensemble import VotingPipeline
from deepmol.pipeline.pipeline import Pipeline
from deepmol.standardizer.basic_standardizer import BasicStandardizer
from tests import TEST_DIR
from tests.integration_tests.dataset.test_dataset import TestDataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from deepchem.models import GraphConvModel

import numpy as np

def model_build(): # num = number of categories
    input_f = layers.Input(shape=(210,))
    
    X = layers.Dense(2048, activation = 'relu')(input_f)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(3072, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.Dropout(0.2)(X)
    output = layers.Dense(1, activation = 'sigmoid')(X)
    model = keras.Model(inputs = input_f, outputs = output)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss=['binary_crossentropy'])
    return model

def model_build_regression(): # num = number of categories
    input_f = layers.Input(shape=(210,))
    
    X = layers.Dense(2048, activation = 'relu')(input_f)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(3072, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.Dropout(0.2)(X)
    output = layers.Dense(1)(X)
    model = keras.Model(inputs = input_f, outputs = output)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss=['mean_squared_error'])
    return model

class TestDatasetModels(TestDataset):


    def _test_model(self, model):
        model.fit(self.small_dataset_to_test_with_invalid)

        dataset_copy = deepcopy(self.small_dataset_to_test_with_invalid)

        y_pred = model.predict_proba(dataset_copy, return_invalid=False)

        self.assertEqual(y_pred.shape[0], 3)

        dataset_copy = deepcopy(self.small_dataset_to_test_with_invalid)

        y_pred = model.predict(dataset_copy, return_invalid=False)

        self.assertEqual(y_pred.shape[0], 3)

        dataset_copy = deepcopy(self.small_dataset_to_test_with_invalid)

        y_pred = model.predict_proba(dataset_copy, return_invalid=True)

        self.assertEqual(y_pred.shape[0], 6)

        dataset_copy = deepcopy(self.small_dataset_to_test_with_invalid)

        y_pred = model.predict(dataset_copy, return_invalid=True)

        self.assertEqual(y_pred.shape[0], 6)
        self.assertEqual(np.sum(np.isnan(y_pred)), 3)

        rows_with_nan = np.where(np.isnan(y_pred))[0]
        np.testing.assert_array_equal(rows_with_nan, np.array([0, 1, 5]))

    def _test_deepchem_model(self, model):
        model.fit(self.small_dataset_to_test_with_invalid)

        y_pred = model.predict_proba(self.small_dataset_to_test_with_invalid, return_invalid=False)

        self.assertEqual(y_pred.shape[0], 4)

        y_pred = model.predict(self.small_dataset_to_test_with_invalid, return_invalid=False)

        self.assertEqual(y_pred.shape[0], 4)

        y_pred = model.predict_proba(self.small_dataset_to_test_with_invalid, return_invalid=True)

        self.assertEqual(y_pred.shape[0], 6)

        y_pred = model.predict(self.small_dataset_to_test_with_invalid, return_invalid=True)

        self.assertEqual(y_pred.shape[0], 6)
        self.assertEqual(np.sum(np.isnan(y_pred)), 2)
        
        rows_with_nan = np.where(np.isnan(y_pred))[0]
        np.testing.assert_array_equal(rows_with_nan, np.array([0, 1]))

    def test_sklearn_return_invalid(self):

        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)
        TwoDimensionDescriptors(n_jobs=-1).featurize(self.small_dataset_to_test_with_invalid, inplace=True)
        
        self._test_model(model)

        # test with the inplace = False
        self.small_dataset_to_test_with_invalid = TwoDimensionDescriptors(n_jobs=-1).featurize(self.small_dataset_to_test_with_invalid, inplace=False)

        self._test_model(model)

        # test with transform
        self.small_dataset_to_test_with_invalid = TwoDimensionDescriptors(n_jobs=-1).fit_transform(self.small_dataset_to_test_with_invalid)
        self._test_model(model)

    def test_keras_return_invalid(self):

        # force cpu usage
        tf.config.set_visible_devices([], 'GPU')

        model = KerasModel(model_builder=model_build(), epochs=10)
        TwoDimensionDescriptors(n_jobs=-1).featurize(self.small_dataset_to_test_with_invalid, inplace=True)
        
        self._test_model(model)

        # test with the inplace = False
        self.small_dataset_to_test_with_invalid = TwoDimensionDescriptors(n_jobs=-1).featurize(self.small_dataset_to_test_with_invalid, inplace=False)

        self._test_model(model)

        # test with transform
        self.small_dataset_to_test_with_invalid = TwoDimensionDescriptors(n_jobs=-1).fit_transform(self.small_dataset_to_test_with_invalid)
        self._test_model(model)


    def test_deepchem_return_invalid(self):

        ConvMolFeat().featurize(self.small_dataset_to_test_with_invalid, inplace=True)

        # graph = GraphConvModel(n_tasks=1, mode='classification')
        model_graph = DeepChemModel(GraphConvModel, n_tasks=1, mode='classification', epochs=10)

        self._test_deepchem_model(model_graph)

        # test with the inplace = False
        self.small_dataset_to_test_with_invalid = ConvMolFeat(n_jobs=-1).featurize(self.small_dataset_to_test_with_invalid, inplace=False)

        self._test_deepchem_model(model_graph)

        # test with transform
        self.small_dataset_to_test_with_invalid = ConvMolFeat(n_jobs=-1).fit_transform(self.small_dataset_to_test_with_invalid)
        self._test_deepchem_model(model_graph)

    
    def test_pipeline_return_invalid(self):
        standardizer = BasicStandardizer()
        featurizer = TwoDimensionDescriptors()
        feature_selector = LowVarianceFS(threshold=0.1)
        rf_model = RandomForestClassifier()
        rf_model = SklearnModel(model=rf_model, model_dir='sklearn_model')

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('feature_selector', feature_selector),
                 ('model', rf_model)]
        
        pipeline = Pipeline(steps=steps, path='test_pipeline/')

        self._test_model(pipeline)

    def test_voting_pipeline_return_invalid(self):
        standardizer = BasicStandardizer()
        featurizer = TwoDimensionDescriptors()
        feature_selector = LowVarianceFS(threshold=0.1)
        rf_model = RandomForestClassifier()
        rf_model = SklearnModel(model=rf_model, model_dir='sklearn_model')

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('feature_selector', feature_selector),
                 ('model', rf_model)]
        
        pipeline1 = Pipeline(steps=steps, path='test_pipeline/')

        steps = [('standardizer_', BasicStandardizer()),
                 ('featurizer_', MorganFingerprint()),
                 ('feature_selector_', LowVarianceFS(threshold=0.1)),
                 ('model_', SklearnModel(model=RandomForestClassifier(), model_dir='sklearn_model2'))]

        pipeline2 = Pipeline(steps=steps, path='test_pipeline2/')
        
        pipeline = VotingPipeline(pipelines=[pipeline1, pipeline2])

        self._test_model(pipeline)

    def test_voting_pipeline_regressor_return_invalid(self):
        standardizer = BasicStandardizer()
        featurizer = TwoDimensionDescriptors()
        feature_selector = LowVarianceFS(threshold=0.1)
        rf_model = RandomForestRegressor()
        rf_model = SklearnModel(model=rf_model, model_dir='sklearn_model')

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('feature_selector', feature_selector),
                 ('model', rf_model)]
        
        pipeline1 = Pipeline(steps=steps, path='test_pipeline/')

        steps = [('standardizer_', BasicStandardizer()),
                 ('featurizer_', MorganFingerprint()),
                 ('feature_selector_', LowVarianceFS(threshold=0.1)),
                 ('model_', SklearnModel(model=RandomForestRegressor(), model_dir='sklearn_model2'))]

        pipeline2 = Pipeline(steps=steps, path='test_pipeline2/')
        
        pipeline = VotingPipeline(pipelines=[pipeline1, pipeline2])

        self.small_dataset_to_test_with_invalid.mode = "regression"

        self._test_model(pipeline)

    def test_voting_pipeline_mixed_regressor_return_invalid(self):
        standardizer = BasicStandardizer()
        featurizer = MorganFingerprint()
        feature_selector = LowVarianceFS(threshold=0.1)
        rf_model = RandomForestRegressor()
        rf_model = SklearnModel(model=rf_model, model_dir='sklearn_model')

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('feature_selector', feature_selector),
                 ('model', rf_model)]
        
        pipeline1 = Pipeline(steps=steps, path='test_pipeline41/')

        steps = [('standardizer_', BasicStandardizer()),
                 ('featurizer_', TwoDimensionDescriptors()),
                 ('model_', KerasModel(model_builder=model_build_regression(), epochs=10, mode="regression"))]

        pipeline2 = Pipeline(steps=steps, path='test_pipeline3/')
        
        pipeline = VotingPipeline(pipelines=[pipeline1, pipeline2])

        self.small_dataset_to_test_with_invalid.mode = "regression"

        self._test_model(pipeline)

    def test_voting_pipeline_mixed_regressor_deepchem_return_invalid(self):
        standardizer = BasicStandardizer()
        featurizer = ConvMolFeat()
        

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('model', DeepChemModel(GraphConvModel, n_tasks=1, mode='regression', epochs=10))]
        
        pipeline1 = Pipeline(steps=steps, path='test_pipeline/')

        steps = [('standardizer_', BasicStandardizer()),
                 ('featurizer_', TwoDimensionDescriptors()),
                 ('model_', KerasModel(model_builder=model_build_regression(), epochs=10, mode="regression"))]

        pipeline2 = Pipeline(steps=steps, path='test_pipeline2/')
        
        pipeline = VotingPipeline(pipelines=[pipeline1, pipeline2])

        self.small_dataset_to_test_with_invalid.mode = "regression"

        self._test_model(pipeline)

    def test_multilabel(self):
        standardizer = BasicStandardizer()
        featurizer = TwoDimensionDescriptors()
        feature_selector = LowVarianceFS(threshold=0.1)
        rf_model = RidgeClassifier()
        rf_model = SklearnModel(model=rf_model, model_dir='sklearn_model', mode=self.multilabel_classification.mode)

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('feature_selector', feature_selector),
                 ('model', rf_model)]
        
        pipeline1 = Pipeline(steps=steps, path='test_pipeline/')

        steps = [('standardizer_', BasicStandardizer()),
                 ('featurizer_', MorganFingerprint()),
                 ('feature_selector_', LowVarianceFS(threshold=0.1)),
                 ('model_', SklearnModel(model=RidgeClassifier(), model_dir='sklearn_model2', mode=self.multilabel_classification.mode))]

        pipeline2 = Pipeline(steps=steps, path='test_pipeline2/')
        
        pipeline = VotingPipeline(pipelines=[pipeline1, pipeline2])

        pipeline.fit(self.multilabel_classification)
        predictions = pipeline.predict(self.multilabel_classification, return_invalid=True)

        self.assertEqual(predictions.shape[0], 100)

        predictions = pipeline.predict(self.multilabel_classification, return_invalid=False)

        self.assertEqual(predictions.shape[0], 99)

        predictions = pipeline.predict_proba(self.multilabel_classification, return_invalid=False)

        self.assertEqual(predictions.shape[0], 99)

        predictions = pipeline.predict_proba(self.multilabel_classification, return_invalid=True)

        self.assertEqual(predictions.shape[0], 100)




        
