import os
import shutil
import warnings
from copy import copy
from datetime import datetime
from random import randint, random
from unittest import TestCase, skip

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score
from sklearn.svm import SVC
from deepmol.base.transformer import DatasetTransformer

from deepmol.loaders import CSVLoader
from deepmol.loggers import Logger
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.splitters import RandomSplitter

from deepmol.pipeline_optimization._featurizer_objectives import _get_featurizer

import tensorflow as tf

from tests import TEST_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

        dataset = os.path.join(multilabel_dataset_path, "tox21_small_.csv")
        loader = CSVLoader(dataset,
                           smiles_field='smiles',
                           id_field='mol_id',
                           labels_fields=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                                          'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'])
        self.multitask_dataset = loader.create_dataset(sep=",")

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

        self.pipeline_path = os.path.join(TEST_DIR, 'pipelines')

        tf.config.set_visible_devices([], 'GPU')

    def tearDown(self) -> None:
        shutil.rmtree(self.pipeline_path)

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
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 5)
        # assert that 3 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 4)

        df = po.trials_dataframe()
        self.assertEqual(len(df), 5)
        df2 = po.trials_dataframe(cols=['number', 'value'])
        self.assertEqual(df2.shape, (5, 2))

        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, po.best_value)

    @skip("This test is too slow to run on CI.")
    def test_pipeline_optimization_with_ensembles(self):

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
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name, n_pipelines_ensemble=5)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 5)
        # assert that 3 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 4)

        df = po.trials_dataframe()
        self.assertEqual(len(df), 5)
        df2 = po.trials_dataframe(cols=['number', 'value'])
        self.assertEqual(df2.shape, (5, 2))

        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, po.best_value)

        results = po.pipelines_ensemble.evaluate(test, [metric])[0][metric.name]
        self.assertAlmostEqual(results, new_predictions, delta=0.15)

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_predictor_gat(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import gat_model_steps

        def objective(trial):
            gat_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            steps = gat_model_steps(trial, gat_kwargs=gat_kwargs)
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_predictor_gcn(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import gcn_model_steps

        def objective(trial):
            gcn_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            steps = gcn_model_steps(trial, gcn_kwargs=gcn_kwargs)
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_predictor_attentive_fp(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import attentive_fp_model_steps

        def objective(trial):
            attentivefp_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            steps = attentive_fp_model_steps(trial, attentive_fp_kwargs=attentivefp_kwargs)
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_predictor_pagtn_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import pagtn_model_steps

        def objective(trial):
            pagtn_model_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            steps = pagtn_model_steps(trial, pagtn_kwargs=pagtn_model_kwargs)
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_predictor_dag_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import dag_model_steps

        def objective(trial):
            dag_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            steps = dag_model_steps(trial, dag_kwargs=dag_kwargs)
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_predictor_graph_conv_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import graph_conv_model_steps

        def objective(trial):
            graph_conv_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            steps = graph_conv_model_steps(trial, graph_conv_kwargs=graph_conv_kwargs)
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_predictor_smiles_to_vec_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import smiles_to_vec_model_steps
        from deepmol.compound_featurization import SmilesSeqFeat

        def objective(trial):
            smiles_to_vec_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            ssf = SmilesSeqFeat()
            ssf.fit_transform(self.dataset_descriptors)
            chat_to_idx = ssf.char_to_idx
            smiles_to_vec_kwargs['char_to_idx'] = chat_to_idx
            steps = smiles_to_vec_model_steps(trial, smiles_to_vec_kwargs=smiles_to_vec_kwargs)
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI")
    def test_predictor_text_cnn_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import text_cnn_model_steps
        from deepmol.pipeline_optimization._utils import prepare_dataset_for_textcnn
        from deepchem.models import TextCNNModel
        
        def objective(trial):
            text_cnn_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            max_length = max([len(smile) for smile in self.dataset_descriptors.smiles])
            padded_train_smiles = prepare_dataset_for_textcnn(self.dataset_descriptors, max_length).ids
            fake_dataset = copy(self.dataset_descriptors)
            fake_dataset._ids = padded_train_smiles
            char_dict, seq_length = TextCNNModel.build_char_dict(fake_dataset)
            text_cnn_kwargs['char_dict'] = char_dict
            text_cnn_kwargs['seq_length'] = seq_length
            padder = DatasetTransformer(prepare_dataset_for_textcnn, max_length=max_length)
            steps = []
            steps.extend([('padder', padder), text_cnn_model_steps(trial=trial, text_cnn_kwargs=text_cnn_kwargs)[0]])
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI")
    def test_predictor_weave_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import weave_model_steps

        def objective(trial):
            weave_kwargs = {'n_tasks': 1, 'mode': "classification", 'n_classes': 2}
            steps = weave_model_steps(trial, weave_kwargs=weave_kwargs)
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI")
    def test_predictor_dtnn_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import dtnn_model_steps
        from deepmol.compound_featurization import CoulombFeat

        def objective(trial):
            max_atoms = max([mol.GetNumAtoms() for mol in self.dataset_descriptors.mols])
            featurizer = CoulombFeat(max_atoms=max_atoms)
            dtnn_kwargs = {'n_tasks': 1}
            model_step = dtnn_model_steps(trial=trial, dtnn_kwargs=dtnn_kwargs)
            final_steps = [('featurizer', featurizer), model_step[0]]
            return final_steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is too slow to run on CI")
    def test_predictor_robust_multitask_classifier_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import robust_multitask_classifier_model_steps

        def objective(trial):
            n_tasks = len(self.multitask_dataset.mode)
            if self.multitask_dataset.mode == 'classification':
                n_classes = len(set(self.multitask_dataset.y))
            else:
                n_classes = len(set(self.multitask_dataset.y[0]))

            featurizer = _get_featurizer(trial, '1D')
            n_features = len(featurizer.feature_names)
            robust_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes}
            model_step = robust_multitask_classifier_model_steps(trial=trial,
                                                                    robust_multitask_classifier_kwargs=robust_multitask_classifier_kwargs)
            featurizer = ('featurizer', featurizer)
            steps = [featurizer, model_step[0]]
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.multitask_dataset, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test is not passing")
    def test_predictor_progressive_multitask_classifier_model_steps(self):
        from deepmol.pipeline_optimization._deepchem_models_objectives import progressive_multitask_classifier_model_steps

        def objective(trial):
            n_tasks = len(self.multitask_dataset.mode)
            if self.multitask_dataset.mode == 'classification':
                n_classes = len(set(self.multitask_dataset.y))
            else:
                n_classes = 2

            featurizer = _get_featurizer(trial, '1D')
            n_features = len(featurizer.feature_names)
            progressive_multitask_classifier_kwargs = {'n_tasks': n_tasks, 'n_features': n_features, 'n_classes': n_classes}
            model_step = progressive_multitask_classifier_model_steps(trial=trial,
                                                                    progressive_multitask_classifier_kwargs=progressive_multitask_classifier_kwargs)
            featurizer = ('featurizer', featurizer)
            steps = [featurizer, model_step[0]]
            return steps

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.multitask_dataset, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=objective, metric=metric, n_trials=5,
                    save_top_n=3)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_sklearn_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_sklearn_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.multitask_dataset, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_sklearn_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_sklearn_classification_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_sklearn_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_sklearn_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_sklearn_regression_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_sklearn_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='minimize', study_name=study_name)
        metric = Metric(mean_squared_error)
        train, test = RandomSplitter().train_test_split(self.dataset_regression, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_sklearn_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_keras_multi_task_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_keras_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.multitask_dataset, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_keras_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)
    
    @skip("This test takes too much time to run on CI")
    def test_keras_classification_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_keras_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_keras_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*1)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_keras_regression_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_keras_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='minimize', study_name=study_name)
        metric = Metric(mean_squared_error)
        train, test = RandomSplitter().train_test_split(self.dataset_regression, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_keras_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_deepchem_multi_task_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_deepchem_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.multitask_dataset, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_deepchem_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_deepchem_classification_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_deepchem_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_deepchem_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_deepchem_regression_model_steps(self):
        from deepmol.pipeline_optimization._utils import preset_deepchem_models

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='minimize', study_name=study_name)
        metric = Metric(mean_squared_error)
        train, test = RandomSplitter().train_test_split(self.dataset_regression, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps=preset_deepchem_models, metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)
    
    @skip
    def test_all_regression_model_steps(self):

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='minimize', study_name=study_name)
        metric = Metric(mean_squared_error)
        train, test = RandomSplitter().train_test_split(self.dataset_regression, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps="all", metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip
    def test_all_classification_model_steps(self):

        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_descriptors, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps="all", metric=metric, n_trials=100,
                    save_top_n=3, data=train, trial_timeout=60*10)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

    @skip("This test takes too much time to run on CI")
    def test_gcn_featurization(self):

        from deepmol.standardizer import CustomStandardizer
        from deepmol.compound_featurization import MolGraphConvFeat
        from deepmol.pipeline import Pipeline

        standardizer = CustomStandardizer()
        featurizer = MolGraphConvFeat(use_chirality=True, use_edges=True, use_partial_charge=True)
        new_pipeline_steps = [("standardizer", standardizer), ("featurizer", featurizer)]
        new_pipeline = Pipeline(new_pipeline_steps)
        new_pipeline.fit_transform(self.multitask_dataset) 

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_classification_preset(self):
        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name, storage=storage_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_smiles, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='deepchem', metric=metric, n_trials=3,
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

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_multiclass_classification_preset(self):
        warnings.filterwarnings("ignore")
        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name, storage=storage_name)
        metric = Metric(accuracy_score)
        train, test = RandomSplitter().train_test_split(self.dataset_multiclass, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='deepchem', metric=metric, n_trials=3,
                    data=train, save_top_n=2, trial_timeout=10*60)
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

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_multi_label_classification_keras(self):
        warnings.filterwarnings("ignore")
        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='maximize', study_name=study_name, storage=storage_name)

        def precision_score_macro(y_true, y_pred):
            return precision_score(y_true, y_pred, average='macro')

        metric = Metric(precision_score_macro)
        train, test = RandomSplitter().train_test_split(self.multitask_dataset, seed=123)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='keras', metric=metric, n_trials=3,
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

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_multilabel_classification_preset(self):
        warnings.filterwarnings("ignore")
        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
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

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_regression_preset(self):
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
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

    @skip("This test is too slow to run on CI and can have different results on different trials")
    def test_multilabel_regression_preset(self):

        storage_name = "sqlite:///test_pipeline.db"
        study_name = f"test_predictor_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study_name = os.path.join(self.pipeline_path, study_name)
        po = PipelineOptimization(direction='minimize', study_name=study_name, storage=storage_name)
        metric = Metric(mean_squared_error)
        train, test = RandomSplitter().train_test_split(self.dataset_multilabel_regression, seed=124)
        po.optimize(train_dataset=train, test_dataset=test, objective_steps='keras', metric=metric, n_trials=3,
                    data=train, save_top_n=2)
        self.assertEqual(po.best_params, po.best_trial.params)
        self.assertIsInstance(po.best_value, float)

        self.assertEqual(len(po.trials), 3)
        # assert that 2 pipelines were saved
        self.assertEqual(len(os.listdir(study_name)), 2)
        best_value = po.best_value
        best_pipeline = po.best_pipeline
        new_predictions = best_pipeline.evaluate(test, [metric])[0][metric.name]
        self.assertEqual(new_predictions, best_value)

        param_importance = po.get_param_importances()
        if param_importance is not None:
            for param in param_importance:
                self.assertTrue(param in po.best_params.keys())
