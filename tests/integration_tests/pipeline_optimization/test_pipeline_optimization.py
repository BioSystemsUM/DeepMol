import os
import shutil
from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.splitters import RandomSplitter

from tests import TEST_DIR


class TestPipelineOptimization(TestCase):

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

        def objective(trial, train_dataset, test_dataset, metric):
            model = trial.suggest_categorical('model', ['RandomForestClassifier', 'SVC'])
            if model == 'RandomForestClassifier':
                n_estimators = trial.suggest_int('model__n_estimators', 10, 100, step=10)
                model = RandomForestClassifier(n_estimators=n_estimators)
            elif model == 'SVC':
                kernel = trial.suggest_categorical('model__kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                model = SVC(kernel=kernel)
            model = SklearnModel(model=model, model_dir='model')
            pipeline = Pipeline(steps=[('model', model)], path='test_predictor_pipeline/')
            pipeline.fit(train_dataset)
            score = pipeline.evaluate(test_dataset, [metric])[0][metric.name]
            return score

        po = PipelineOptimization(direction='maximize')
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective=objective, metric=metric, n_trials=5)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    def test_preset(self):
        po = PipelineOptimization(direction='maximize')
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_smiles, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective='ss', metric=metric, n_trials=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)
