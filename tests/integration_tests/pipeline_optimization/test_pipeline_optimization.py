import os
import shutil
import warnings
from copy import copy
from datetime import datetime
from random import randint, random
from unittest import TestCase

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC

from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.splitters import RandomSplitter

from tests import TEST_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestPipelineOptimization(TestCase):

    def setUp(self) -> None:
        warnings.filterwarnings("ignore")
        data_descriptors_path = os.path.join(TEST_DIR, 'data')
        dataset_descriptors = os.path.join(data_descriptors_path, "small_train_dataset.csv")
        features = [f'feat_{i}' for i in range(1, 2049)]
        loader = CSVLoader(dataset_descriptors,
                           smiles_field='mols',
                           id_field='ids',
                           labels_fields=['y'],
                           features_fields=features,
                           mode='classification',
                           shard_size=100)
        self.dataset_descriptors = loader.create_dataset(sep=",")

        smiles_dataset_path = os.path.join(TEST_DIR, 'data')
        dataset_smiles = os.path.join(smiles_dataset_path, "balanced_mini_dataset.csv")
        loader = CSVLoader(dataset_smiles,
                           smiles_field='Smiles',
                           labels_fields=['Class'],
                           mode='classification')
        self.dataset_smiles = loader.create_dataset(sep=";")

        regression_dataset_path = os.path.join(TEST_DIR, 'data')
        dataset_regression = os.path.join(regression_dataset_path, "PC-3.csv")
        loader = CSVLoader(dataset_regression,
                           smiles_field='smiles',
                           labels_fields=['pIC50'],
                           mode='regression',
                           shard_size=100)
        self.dataset_regression = loader.create_dataset(sep=",")

        self.dataset_multiclass = copy(self.dataset_smiles)
        # change dataset.y to multiclass
        self.dataset_multiclass._y = np.array([randint(0, 3) for _ in range(len(self.dataset_multiclass))])

        multilabel_dataset_path = os.path.join(TEST_DIR, 'data')
        dataset_multilabel = os.path.join(multilabel_dataset_path, "multilabel_classification_dataset.csv")
        loader = CSVLoader(dataset_multilabel,
                           smiles_field='smiles',
                           labels_fields=['C00341', 'C01789', 'C00078'],
                           mode=['classification', 'classification', 'classification'],
                           shard_size=100)
        self.dataset_multilabel = loader.create_dataset(sep=",")

        self.dataset_multilabel_regression = copy(self.dataset_multilabel)
        # change dataset.y to regression (3 labels with random float values between 0 and 10)
        self.dataset_multilabel_regression._y = np.array([[random() * 10 for _ in range(3)] for _ in
                                                          range(len(self.dataset_multilabel_regression))])
        self.dataset_multilabel_regression.mode = ['regression', 'regression', 'regression']

    def tearDown(self) -> None:
        # remove logs (files with .log extension)
        for file in os.listdir():
            if file.endswith('.log'):
                os.remove(file)

        # remove model directories (ending with _model)
        for file in os.listdir():
            if file.endswith('_model'):
                shutil.rmtree(file)
            elif file.startswith('test_pipeline_'):
                shutil.rmtree(file)
            elif file.startswith('test_predictor_pipeline_'):
                shutil.rmtree(file)
        # remove study rdbm
        try:
            optuna.delete_study(study_name="test_pipeline", storage="sqlite:///test_pipeline.db")
        except Exception as e:
            print(e)
        # remove db files
        for file in os.listdir():
            if file.endswith('.db'):
                os.remove(file)

    def test_predictor_pipeline(self):

        def objective(trial):
            model = trial.suggest_categorical('model', ['RandomForestClassifier', 'SVC'])
            if model == 'RandomForestClassifier':
                n_estimators = trial.suggest_int('model__n_estimators', 10, 100, step=10)
                model = RandomForestClassifier(n_estimators=n_estimators)
            elif model == 'SVC':
                kernel = trial.suggest_categorical('model__kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                model = SVC(kernel=kernel)
            model = SklearnModel(model=model, model_dir='model')
            steps = [('model', model)]
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 5)
        # assert that 3 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 3)

        df = po.trials_dataframe()
        self.assertEqual(len(df), 5)
        df2 = po.trials_dataframe(cols=['number', 'value'])
        self.assertEqual(df2.shape, (5, 2))

    def test_binary_classification_preset(self):
        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        po = PipelineOptimization(direction='maximize', study_name=study_name, storage=storage_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_smiles, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='all', metric=metric, n_trials=3,
                    data=train, save_top_n=2)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 3)
        # assert that 2 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 2)

        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, po.best_value)

        param_importance = po.get_param_importances()
        if param_importance is not None:
            for param in param_importance:
                self.assertTrue(param in po.best_params.keys())

    def test_multiclass_classification_preset(self):
        warnings.filterwarnings("ignore")
        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        po = PipelineOptimization(direction='maximize', study_name=study_name, storage=storage_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_multiclass, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='all', metric=metric, n_trials=3,
                    data=train, save_top_n=2)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 3)
        # assert that 2 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 2)

        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, po.best_value)

        param_importance = po.get_param_importances()
        if param_importance is not None:
            for param in param_importance:
                self.assertTrue(param in po.best_params.keys())

    def test_multilabel_classification_preset(self):
        warnings.filterwarnings("ignore")
        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        po = PipelineOptimization(direction='maximize', study_name=study_name, storage=storage_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_multilabel, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='all', metric=metric, n_trials=3,
                    data=train, save_top_n=1)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 3)
        # assert that 2 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 1)
        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, po.best_value)

        param_importance = po.get_param_importances()
        if param_importance is not None:
            for param in param_importance:
                self.assertTrue(param in po.best_params.keys())

    def test_regression_preset(self):
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        po = PipelineOptimization(direction='minimize', study_name=study_name)
        metric = Metric(mean_squared_error)
        train, test = RandomSplitter().train_test_split(self.dataset_regression, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='all', metric=metric, n_trials=3,
                    data=train, save_top_n=2)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 3)
        # assert that 2 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 2)

        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, po.best_value)

    def test_multilabel_regression_preset(self):
        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        po = PipelineOptimization(direction='minimize', study_name=study_name, storage=storage_name)
        metric = Metric(mean_squared_error)
        train, test = RandomSplitter().train_test_split(self.dataset_multilabel_regression, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='all', metric=metric, n_trials=3,
                    data=train, save_top_n=2)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 3)
        # assert that 2 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 2)
        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, po.best_value)

        param_importance = po.get_param_importances()
        if param_importance is not None:
            for param in param_importance:
                self.assertTrue(param in po.best_params.keys())
