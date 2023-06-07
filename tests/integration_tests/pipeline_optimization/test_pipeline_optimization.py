import os
import shutil
from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
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

        po = PipelineOptimization(direction='maximize', study_name='test_predictor_pipeline')
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 5)
        # assert that 3 pipelines were saved
        self.assertEqual(len(os.listdir('test_predictor_pipeline')), 3)

    def test_preset(self):
        po = PipelineOptimization(direction='maximize', study_name='test_pipeline')
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_smiles, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='ss', metric=metric, n_trials=3, data=train,
                    save_top_n=2)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 3)
        # assert that 2 pipelines were saved
        self.assertEqual(len(os.listdir('test_pipeline')), 2)

        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, po.best_value)
