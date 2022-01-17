import os
from unittest import TestCase

from deepchem.models import TextCNNModel
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from compoundFeaturization import deepChemFeaturizers
from loaders.Loaders import CSVLoader
from metrics.Metrics import Metric
from metrics.metricsFunctions import prc_auc_score
from models.DeepChemModels import DeepChemModel
from parameterOptimization.HyperparameterOpt import HyperparamOpt_CV


class TestDeepChemHyperparameterOpt(TestCase):

    def setUp(self) -> None:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(".")))
        dataset = os.path.join(dir_path, "tests", "data", "train_dataset.csv")

        loader = CSVLoader(dataset,
                           mols_field='mols',
                           labels_fields='y')
        self.dataset = loader.create_dataset()

    def test_hyperparameter_opt(self):
        def textcnn_builder(char_dict, seq_length, n_embedding, kernel_sizes, num_filters, dropout, learning_rate,
                            task_type,
                            batch_size=256, epochs=100):
            textcnn = TextCNNModel(n_tasks=1, char_dict=char_dict, seq_length=seq_length, n_embedding=n_embedding,
                                   kernel_sizes=kernel_sizes, num_filters=num_filters, dropout=dropout,
                                   batch_size=batch_size, learning_rate=learning_rate, mode=task_type)
            return DeepChemModel(textcnn, epochs=epochs, use_weights=False, model_dir=None)

        params_dict = {'n_embedding': [75, 32, 64],
                       'kernel_sizes': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                                        # DeepChem default. Their code says " Multiple convolutional layers with different filter widths", so I'm not repeating kernel_sizes
                                        [1, 2, 3, 4, 5, 7, 10, 15],
                                        [3, 4, 5, 7, 10, 15],
                                        [3, 4, 5, 7, 10],
                                        [3, 4, 5, 7],
                                        [3, 4, 5],
                                        [3, 5, 7]],
                       'num_filters': [[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],  # DeepChem default
                                       [32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128],
                                       [128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32]],
                       'dropout': [0.0, 0.25, 0.5],
                       'learning_rate': [1e-4, 1e-3, 1e-2]}

        featurizer = deepChemFeaturizers.RawFeat()
        featurizer.featurize(self.dataset)

        self.dataset.ids = self.dataset.mols
        char_dict, length = TextCNNModel.build_char_dict(self.dataset)
        params_dict['char_dict'] = [char_dict]
        params_dict['seq_length'] = [length]
        mode = 'classification'
        params_dict['task_type'] = mode


        optimizer = HyperparamOpt_CV(textcnn_builder, mode=mode)

        metrics = [Metric(roc_auc_score, n_tasks=1),
                   Metric(prc_auc_score, n_tasks=1),
                   Metric(accuracy_score, n_tasks=1),
                   Metric(precision_score, n_tasks=1),
                   Metric(recall_score, n_tasks=1)]
        opt_metric = 'roc_auc'

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search("deepchem",
                                                                                params_dict,
                                                                                self.dataset,
                                                                                opt_metric,
                                                                                cv=3,
                                                                                n_iter_search=1,
                                                                                n_jobs=1,
                                                                                seed=123)
