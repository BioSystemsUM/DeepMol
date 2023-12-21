import os
import shutil
from unittest import TestCase

import numpy as np
from deepchem.models import GraphConvModel
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from deepmol.base import PassThroughTransformer
from deepmol.compound_featurization import MorganFingerprint, LayeredFingerprint, ConvMolFeat
from deepmol.feature_selection import LowVarianceFS, KbestFS
from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.models import SklearnModel, KerasModel, DeepChemModel
from deepmol.parameter_optimization import HyperparameterOptimizerValidation, HyperparameterOptimizerCV
from deepmol.pipeline import Pipeline
from deepmol.scalers import StandardScaler, MinMaxScaler
from deepmol.standardizer import BasicStandardizer, ChEMBLStandardizer
from deepmol.unsupervised import PCA
from tests import TEST_DIR


class TestPipeline(TestCase):

    def setUp(self) -> None:
        data_descriptors_path = os.path.join(TEST_DIR, 'data')
        dataset_descriptors = os.path.join(data_descriptors_path, "small_train_dataset.csv")
        features = [f'feat_{i}' for i in range(1, 2049)]
        loader = CSVLoader(dataset_descriptors,
                           smiles_field='mols',
                           id_field='ids',
                           labels_fields=['y'],
                           features_fields=features)
        self.dataset_descriptors = loader.create_dataset(sep=",")

        smiles_dataset_path = os.path.join(TEST_DIR, 'data')
        dataset_smiles = os.path.join(smiles_dataset_path, "balanced_mini_dataset.csv")
        loader = CSVLoader(dataset_smiles,
                           smiles_field='Smiles',
                           labels_fields=['Class'])
        self.dataset_smiles = loader.create_dataset(sep=";")

    def tearDown(self) -> None:
        if os.path.exists('test_predictor_pipeline'):
            shutil.rmtree('test_predictor_pipeline')
        if os.path.exists('test_pipeline'):
            shutil.rmtree('test_pipeline')
        # remove logs (files starting with 'deepmol.log')
        for f in os.listdir():
            if f.startswith('deepmol.log'):
                os.remove(f)

    def test_predictor_pipeline(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf, model_dir='model')
        pipeline = Pipeline(steps=[('model', model)], path='test_predictor_pipeline/')

        pipeline.save()

        pipeline2 = Pipeline.load(pipeline.path)
        self.assertFalse(pipeline2.is_fitted())

        pipeline.fit_transform(self.dataset_descriptors)
        predictions = pipeline.predict(self.dataset_descriptors)
        self.assertEqual(len(predictions), len(self.dataset_descriptors))
        e1, e2 = pipeline.evaluate(self.dataset_descriptors, [Metric(accuracy_score)])
        self.assertTrue('accuracy_score' in e1.keys())
        self.assertEqual(e2, {})

        pipeline.save()

        pipeline3 = Pipeline.load(pipeline.path)
        self.assertTrue(pipeline3.is_fitted())

        predictions_p3 = pipeline3.predict(self.dataset_descriptors)
        self.assertEqual(len(predictions_p3), len(self.dataset_descriptors))
        for pred, pred_p3 in zip(predictions, predictions_p3):
            self.assertTrue(np.array_equal(pred, pred_p3))
        e1_p3, e2_p3 = pipeline3.evaluate(self.dataset_descriptors, [Metric(accuracy_score)])
        self.assertEqual(e1, e1_p3)
        self.assertEqual(e2_p3, {})

    def test_hpo_validation_sklearn_models(self):
        standardizer = BasicStandardizer()
        featurizer = MorganFingerprint(size=1024)
        feature_selector = LowVarianceFS(threshold=0.1)
        svc_model = SVC()
        svc_model = SklearnModel(model=svc_model, model_dir='sklearn_model')

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('feature_selector', feature_selector),
                 ('model', svc_model)]

        params_dict_svc = {"C": [1.0, 1.2, 0.8]}

        hpo = HyperparameterOptimizerValidation(SVC, metric=Metric(accuracy_score),
                                                maximize_metric=True,
                                                n_iter_search=2,
                                                params_dict=params_dict_svc,
                                                model_type="sklearn")

        pipeline = Pipeline(steps=steps, hpo=hpo, path='test_pipeline/')
        pipeline.fit(self.dataset_smiles, self.dataset_smiles)
        self.assertTrue(pipeline.is_fitted())
        self.assertTrue(len(pipeline.hpo_all_results_) == 2)
        self.assertTrue(len(pipeline.hpo_best_hyperparams_) == 1)

        self.assertEqual(pipeline.predict(self.dataset_smiles).shape, self.dataset_smiles.y.shape)
        self.assertEqual(pipeline.steps[-1][0], "best_model")

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('feature_selector', feature_selector)]
        pipeline = Pipeline(steps=steps, hpo=hpo, path='test_pipeline/')
        pipeline.fit(self.dataset_smiles, self.dataset_smiles)
        self.assertTrue(pipeline.is_fitted())
        self.assertTrue(len(pipeline.hpo_all_results_) == 2)
        self.assertTrue(len(pipeline.hpo_best_hyperparams_) == 1)
        self.assertEqual(pipeline.steps[-1][0], "best_model")

        predictions = pipeline.predict(self.dataset_smiles)
        pipeline.save()
        pipeline = Pipeline.load(pipeline.path)
        new_predictions = pipeline.predict(self.dataset_smiles)
        self.assertTrue(np.array_equal(predictions, new_predictions))

    def test_hpo_cv_sklearn_models(self):
        standardizer = BasicStandardizer()
        featurizer = MorganFingerprint(size=1024)
        feature_selector = LowVarianceFS(threshold=0.1)
        svc_model = SVC()
        svc_model = SklearnModel(model=svc_model, model_dir='sklearn_model')

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('feature_selector', feature_selector),
                 ('model', svc_model)]

        params_dict_svc = {"C": [1.0, 1.2, 0.8]}

        hpo = HyperparameterOptimizerCV(SVC, metric=Metric(accuracy_score),
                                        maximize_metric=True,
                                        n_iter_search=2,
                                        cv=3,
                                        params_dict=params_dict_svc,
                                        model_type="sklearn")

        pipeline = Pipeline(steps=steps, hpo=hpo, path='test_pipeline/')
        pipeline.fit(self.dataset_smiles)

        self.assertTrue(pipeline.is_fitted())
        self.assertTrue(len(pipeline.hpo_all_results_) == 12)
        self.assertTrue(len(pipeline.hpo_best_hyperparams_) == 1)
        self.assertEqual(pipeline.steps[-1][0], "best_model")
        predictions = pipeline.predict(self.dataset_smiles)
        pipeline.save()
        pipeline = Pipeline.load(pipeline.path)
        new_predictions = pipeline.predict(self.dataset_smiles)
        self.assertTrue(np.array_equal(predictions, new_predictions))

    def test_hpo_validation_keras_models(self):
        def basic_classification_model_builder(input_shape, layers):
            model = Sequential()
            model.add(Input(shape=input_shape))
            for layer in layers:
                model.add(Dense(layer, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        keras_model = KerasModel(model_builder=basic_classification_model_builder, epochs=2, input_shape=(1024,))
        standardizer = BasicStandardizer()
        featurizer = MorganFingerprint(size=1024)

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('model', keras_model)]

        params_dict_keras = {"layers": [[512, 256], [512, 256, 128]], "input_shape": [(1024,)]}
        optimizer = HyperparameterOptimizerValidation(basic_classification_model_builder, metric=Metric(accuracy_score),
                                                      maximize_metric=True,
                                                      n_iter_search=2,
                                                      params_dict=params_dict_keras,
                                                      model_type="keras", epochs=2)

        pipeline = Pipeline(steps=steps, hpo=optimizer, path='test_pipeline/')
        pipeline.fit(self.dataset_smiles, self.dataset_smiles)
        self.assertTrue(pipeline.is_fitted())
        self.assertTrue(len(pipeline.hpo_all_results_) == 2)
        self.assertTrue(len(pipeline.hpo_best_hyperparams_) == 2)
        self.assertEqual(pipeline.steps[-1][0], "best_model")
        predictions = pipeline.predict(self.dataset_smiles)
        pipeline.save()
        pipeline = Pipeline.load(pipeline.path)
        new_predictions = pipeline.predict(self.dataset_smiles)
        self.assertTrue(np.array_equal(predictions, new_predictions))

    def test_hpo_cv_keras_models(self):
        def basic_classification_model_builder(input_shape, layers):
            model = Sequential()
            model.add(Input(shape=input_shape))
            for layer in layers:
                model.add(Dense(layer, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        keras_model = KerasModel(model_builder=basic_classification_model_builder, epochs=2, input_shape=(1024,))
        standardizer = BasicStandardizer()
        featurizer = MorganFingerprint(size=1024)

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('model', keras_model)]

        params_dict_keras = {"layers": [[512, 256], [512, 256, 128]], "input_shape": [(1024,)]}
        optimizer = HyperparameterOptimizerCV(basic_classification_model_builder, metric=Metric(accuracy_score),
                                              maximize_metric=True,
                                              n_iter_search=2,
                                              cv=3,
                                              params_dict=params_dict_keras,
                                              model_type="keras", epochs=2)
        pipeline = Pipeline(steps=steps, hpo=optimizer, path='test_pipeline/')
        pipeline.fit(self.dataset_smiles, self.dataset_smiles)
        self.assertTrue(pipeline.is_fitted())
        self.assertTrue(len(pipeline.hpo_all_results_) == 13)
        self.assertTrue(len(pipeline.hpo_best_hyperparams_) == 2)
        self.assertEqual(pipeline.steps[-1][0], "best_model")
        predictions = pipeline.predict(self.dataset_smiles)
        pipeline.save()
        pipeline = Pipeline.load(pipeline.path)
        new_predictions = pipeline.predict(self.dataset_smiles)
        self.assertTrue(np.array_equal(predictions, new_predictions))

    def test_hpo_validation_deepchem_models(self):
        standardizer = BasicStandardizer()
        featurizer = ConvMolFeat()

        def graphconv_builder(graph_conv_layers, batch_size=256, epochs=5):
            return DeepChemModel(GraphConvModel, model_dir=None, epochs=epochs,
                                 n_tasks=1, graph_conv_layers=graph_conv_layers, batch_size=batch_size,
                                   mode='classification')

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer)]

        params_dict_deepchem = {"graph_conv_layers": [[64, 64], [128, 128]]}
        optimizer = HyperparameterOptimizerValidation(graphconv_builder, metric=Metric(accuracy_score),
                                                      maximize_metric=True,
                                                      n_iter_search=2,
                                                      params_dict=params_dict_deepchem,
                                                      model_type="deepchem", epochs=2)

        pipeline = Pipeline(steps=steps, hpo=optimizer, path='test_pipeline/')
        pipeline.fit(self.dataset_smiles, self.dataset_smiles)
        self.assertTrue(pipeline.is_fitted())
        self.assertTrue(len(pipeline.hpo_all_results_) == 2)
        self.assertTrue(len(pipeline.hpo_best_hyperparams_) == 1)
        self.assertEqual(pipeline.steps[-1][0], "best_model")
        predictions = pipeline.predict(self.dataset_smiles)
        pipeline.save()
        pipeline = Pipeline.load(pipeline.path)
        new_predictions = pipeline.predict(self.dataset_smiles)
        self.assertTrue(np.array_equal(predictions, new_predictions))

    def test_hpo_cv_deepchem_models(self):
        standardizer = BasicStandardizer()
        featurizer = ConvMolFeat()

        def graphconv_builder(graph_conv_layers, batch_size=256, epochs=5):
            return DeepChemModel(GraphConvModel, model_dir=None, epochs=epochs,
                                 n_tasks=1, graph_conv_layers=graph_conv_layers, batch_size=batch_size,
                                   mode='classification')

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer)]

        params_dict_deepchem = {"graph_conv_layers": [[64, 64], [128, 128]]}
        optimizer = HyperparameterOptimizerCV(graphconv_builder, metric=Metric(accuracy_score),
                                              maximize_metric=True,
                                              n_iter_search=2,
                                              cv=3,
                                              params_dict=params_dict_deepchem,
                                              model_type="deepchem", epochs=2)

        pipeline = Pipeline(steps=steps, hpo=optimizer, path='test_pipeline/')
        pipeline.fit(self.dataset_smiles)
        self.assertTrue(pipeline.is_fitted())
        self.assertTrue(len(pipeline.hpo_all_results_) == 11)
        self.assertTrue(len(pipeline.hpo_best_hyperparams_) == 1)
        self.assertEqual(pipeline.steps[-1][0], "best_model")
        predictions = pipeline.predict(self.dataset_smiles)
        pipeline.save()
        pipeline = Pipeline.load(pipeline.path)
        new_predictions = pipeline.predict(self.dataset_smiles)
        self.assertTrue(np.array_equal(predictions, new_predictions))

    def test_pipeline(self):
        morgan = MorganFingerprint(size=1024)
        svc = SVC()
        scv = SklearnModel(model=svc, model_dir='model')
        pipeline = Pipeline(steps=[('featurizer', morgan), ('model', scv)], path='test_pipeline/')

        pipeline.fit_transform(self.dataset_smiles)
        predictions = pipeline.predict(self.dataset_smiles)
        self.assertEqual(len(predictions), len(self.dataset_smiles))
        e1, e2 = pipeline.evaluate(self.dataset_smiles, [Metric(accuracy_score)])
        self.assertTrue('accuracy_score' in e1.keys())
        self.assertEqual(e2, {})

        pipeline.save()

        pipeline2 = Pipeline.load(pipeline.path)
        self.assertTrue(pipeline2.is_fitted())

        predictions_p2 = pipeline2.predict(self.dataset_smiles)
        self.assertEqual(len(predictions_p2), len(self.dataset_smiles))
        for pred, pred_p2 in zip(predictions, predictions_p2):
            self.assertTrue(np.array_equal(pred, pred_p2))
        e1_p2, e2_p2 = pipeline2.evaluate(self.dataset_smiles, [Metric(accuracy_score)])
        self.assertEqual(e1, e1_p2)
        self.assertEqual(e2_p2, {})

    def validate_complete_pipeline(self, standardizer, featurizer, scaler, feature_selector, unsupervised, model):
        standardizer = standardizer
        featurizer = featurizer
        feature_selector = feature_selector
        model = model

        steps = [('standardizer', standardizer),
                 ('featurizer', featurizer),
                 ('scaler', scaler),
                 ('feature_selector', feature_selector),
                 ('unsupervised', unsupervised),
                 ('model', model)]

        # pop steps that are None
        steps = [step for step in steps if step[1] is not None]

        pipeline = Pipeline(steps=steps, path='test_pipeline/')

        pipeline.fit_transform(self.dataset_smiles)
        predictions = pipeline.predict(self.dataset_smiles)
        self.assertEqual(len(predictions), len(self.dataset_smiles))
        e1, e2 = pipeline.evaluate(self.dataset_smiles, [Metric(accuracy_score)])
        self.assertTrue('accuracy_score' in e1.keys())
        self.assertEqual(e2, {})

        pipeline.save()

        pipeline2 = Pipeline.load(pipeline.path)
        self.assertTrue(pipeline2.is_fitted())

        predictions_p2 = pipeline2.predict(self.dataset_smiles)
        self.assertEqual(len(predictions_p2), len(self.dataset_smiles))
        for pred, pred_p2 in zip(predictions, predictions_p2):
            self.assertTrue(np.array_equal(pred, pred_p2))
        e1_p2, e2_p2 = pipeline2.evaluate(self.dataset_smiles, [Metric(accuracy_score)])
        self.assertEqual(e1, e1_p2)
        self.assertEqual(e2_p2, {})

    def test_complete_pipeline(self):
        svc_model = SVC()
        svc_model = SklearnModel(model=svc_model, model_dir='sklearn_model')

        self.validate_complete_pipeline(standardizer=BasicStandardizer(),
                                        featurizer=MorganFingerprint(size=1024),
                                        scaler=StandardScaler(),
                                        feature_selector=LowVarianceFS(threshold=0.1),
                                        unsupervised=PCA(n_components=2),
                                        model=svc_model)

        def basic_classification_model_builder(input_shape):
            model = Sequential()
            model.add(Input(shape=input_shape))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        keras_model = KerasModel(model_builder=basic_classification_model_builder, epochs=2, input_shape=(10,))
        self.validate_complete_pipeline(standardizer=ChEMBLStandardizer(),
                                        featurizer=LayeredFingerprint(fpSize=1024),
                                        scaler=MinMaxScaler(feature_range=(0, 1)),
                                        feature_selector=KbestFS(k=10),
                                        unsupervised=None,
                                        model=keras_model)

        deepchem_model = GraphConvModel(n_tasks=1, mode='classification')
        model_graph = DeepChemModel(deepchem_model)
        self.validate_complete_pipeline(standardizer=ChEMBLStandardizer(),
                                        featurizer=ConvMolFeat(),
                                        scaler=PassThroughTransformer(),
                                        feature_selector=None,
                                        unsupervised=None,
                                        model=model_graph)
