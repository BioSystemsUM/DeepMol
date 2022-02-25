import os
from unittest import TestCase

from deepchem.models import TextCNNModel, GraphConvModel, torch_models, ChemCeption
from deepchem.models.chemnet_models import DEFAULT_INCEPTION_BLOCKS
from deepchem.models.layers import DTNNEmbedding, Highway
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    classification_report, f1_score

from compound_featurization import deepchem_featurizers
from loaders.loaders import CSVLoader
from metrics.metrics import Metric
from metrics.metrics_functions import prc_auc_score
from models.deepchem_models import DeepChemModel
from models.keras_models import KerasModel
from parameter_optimization.hyperparameter_optimization import HyperparamOpt_CV


class TestDeepChemHyperparameterOpt(TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.abspath("."))
        dataset = os.path.join(dir_path, "tests", "data", "train_dataset.csv")

        loader = CSVLoader(dataset,
                           mols_field='mols',
                           labels_fields='y')
        self.dataset = loader.create_dataset()

    def test_hyperparameter_opt(self):
        def textcnn_builder(char_dict, seq_length, n_embedding, kernel_sizes, num_filters, dropout, learning_rate,
                            task_type,
                            batch_size=256, epochs=1):
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
        params_dict['task_type'] = [mode]

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
        best_model.evaluate(self.dataset, metrics)

    def test_graph_conv(self):
        def graphconv_builder(graph_conv_layers, dense_layer_size, dropout, learning_rate, task_type, batch_size=256,
                              epochs=100):
            graph = GraphConvModel(n_tasks=1, graph_conv_layers=graph_conv_layers, dense_layer_size=dense_layer_size,
                                   dropout=dropout, batch_size=batch_size, learning_rate=learning_rate, mode=task_type)
            return DeepChemModel(graph, epochs=epochs, use_weights=False, model_dir=None)

        params_dict = {'graph_conv_layers': [[32, 32], [64, 64], [128, 128],
                                             [32, 32, 32], [64, 64, 64], [128, 128, 128],
                                             [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128]],
                       'dense_layer_size': [2048, 1024, 512, 256, 128, 64, 32],
                       'dropout': [0.0, 0.25, 0.5],
                       'learning_rate': [1e-4, 1e-3, 1e-2]}

        mode = 'classification'
        params_dict['task_type'] = [mode]
        params_dict['epochs'] = [1]

        featurizer = deepChemFeaturizers.ConvMolFeat()
        featurizer.featurize(self.dataset)

        optimizer = HyperparamOpt_CV(graphconv_builder, mode=mode)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(f1_score), Metric(classification_report)]

        opt_metric = 'roc_auc'

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search("deepchem",
                                                                                params_dict,
                                                                                self.dataset,
                                                                                opt_metric,
                                                                                cv=3,
                                                                                n_iter_search=1,
                                                                                n_jobs=1,
                                                                                seed=123)

    def test_gcn(self):
        def gcn_builder(graph_conv_layers, dropout, predictor_hidden_feats, predictor_dropout, learning_rate, task_type,
                        batch_size=256, epochs=100):
            gcn = torch_models.GCNModel(n_tasks=1, graph_conv_layers=graph_conv_layers, activation=None,
                                        residual=True, batchnorm=False, predictor_hidden_feats=predictor_hidden_feats,
                                        dropout=dropout, predictor_dropout=predictor_dropout,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size, mode=task_type)
            return DeepChemModel(gcn, epochs=epochs, use_weights=False, model_dir=None)

        params_dict = {'graph_conv_layers': [[32, 32], [64, 64], [128, 128],
                                             [32, 32, 32], [64, 64, 64], [128, 128, 128],
                                             [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128]],
                       'predictor_hidden_feats': [256, 128, 64],
                       'dropout': [0.0, 0.25, 0.5],
                       'predictor_dropout': [0.0, 0.25, 0.5],
                       'learning_rate': [1e-4, 1e-3, 1e-2]}

        mode = 'classification'
        params_dict['task_type'] = [mode]
        params_dict['epochs'] = [1]

        featurizer = deepChemFeaturizers.MolGraphConvFeat()
        featurizer.featurize(self.dataset)

        optimizer = HyperparamOpt_CV(gcn_builder, mode=mode)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(f1_score), Metric(classification_report)]

        opt_metric = 'roc_auc'

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search("deepchem",
                                                                                params_dict,
                                                                                self.dataset,
                                                                                opt_metric,
                                                                                cv=3,
                                                                                n_iter_search=1,
                                                                                n_jobs=1,
                                                                                seed=123)

    def test_GAT(self):
        def gat_builder(graph_attention_layers, n_attention_heads, dropout, predictor_hidden_feats,
                        predictor_dropout, learning_rate, task_type, batch_size=256, epochs=100):
            gat = torch_models.GATModel(n_tasks=1, graph_attention_layers=graph_attention_layers,
                                        n_attention_heads=n_attention_heads, dropout=dropout,
                                        predictor_hidden_feats=predictor_hidden_feats,
                                        predictor_dropout=predictor_dropout,
                                        learning_rate=learning_rate, batch_size=batch_size, mode=task_type)
            return DeepChemModel(gat, epochs=epochs, use_weights=False, model_dir=None)

        params_dict = {'graph_attention_layers': [[8, 8], [16, 16], [32, 32],
                                                  [8, 8, 8], [16, 16, 16], [32, 32, 32]],
                       'n_attention_heads': [4, 8],
                       'predictor_hidden_feats': [256, 128, 64],
                       'dropout': [0.0, 0.25, 0.5],
                       'predictor_dropout': [0.0, 0.25, 0.5],
                       'learning_rate': [1e-4, 1e-3, 1e-2]}

        mode = 'classification'
        params_dict['task_type'] = [mode]
        params_dict['epochs'] = [1]

        featurizer = deepChemFeaturizers.MolGraphConvFeat()
        featurizer.featurize(self.dataset)

        optimizer = HyperparamOpt_CV(gat_builder, mode=mode)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(f1_score), Metric(classification_report)]

        opt_metric = 'roc_auc'

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search("deepchem",
                                                                                params_dict,
                                                                                self.dataset,
                                                                                opt_metric,
                                                                                cv=3,
                                                                                n_iter_search=1,
                                                                                n_jobs=1,
                                                                                seed=123)

    def test_chemception(self):
        def chemception_builder(img_spec, img_size, base_filters, inception_blocks,
                                n_classes, augment, mode, learning_rate, batch_size=256, epochs=100):
            chemception = ChemCeption(img_spec=img_spec, img_size=img_size, base_filters=base_filters,
                                      inception_blocks=inception_blocks, n_tasks=1,
                                      n_classes=n_classes, augment=augment, mode=mode,
                                      learning_rate=learning_rate, batch_size=batch_size)

            return DeepChemModel(chemception, epochs=epochs, use_weights=False, model_dir=None)

        params_dict = {'img_spec': ["std"],
                       'img_size': [80],
                       'base_filters': [8, 16, 32, 64],
                       'inception_blocks': [{"A": 4, "B": 4, "C": 4}, {"A": 8, "B": 8, "C": 8},
                                            {"A": 16, "B": 16, "C": 16}, {"A": 32, "B": 32, "C": 32}],
                       'augment': [True, False],
                       'learning_rate': [1e-4, 1e-3, 1e-2]}

        mode = 'classification'
        img_size = 80
        img_spec = "std"
        params_dict['n_classes'] = [2]
        params_dict['mode'] = [mode]
        params_dict['img_size'] = [img_size]
        params_dict['img_spec'] = [img_spec]
        params_dict['epochs'] = [1]

        featurizer = deepChemFeaturizers.SmileImageFeat()
        featurizer.featurize(self.dataset)

        optimizer = HyperparamOpt_CV(chemception_builder, mode=mode)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(f1_score), Metric(classification_report)]

        opt_metric = 'roc_auc'

        best_model, best_hyperparams, all_results = optimizer.hyperparam_search("deepchem",
                                                                                params_dict,
                                                                                self.dataset,
                                                                                opt_metric,
                                                                                cv=3,
                                                                                n_iter_search=1,
                                                                                n_jobs=1,
                                                                                seed=123)

    def test_quick(self):

        from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization, Activation
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l1_l2
        import tensorflow as tf

        def dense_builder(input_dim=None, hlayers_sizes=[10], initializer='he_normal',
                          l1=0, l2=0, dropout=0, batchnorm=True, learning_rate_value=0.001,
                          optimizer_name="Adam", batch_size=200):

            model = Sequential()
            model.add(Input(shape=input_dim))

            for i in range(len(hlayers_sizes)):
                model.add(Dense(units=hlayers_sizes[i], kernel_initializer=initializer,
                                kernel_regularizer=l1_l2(l1=l1, l2=l2)))
                if batchnorm:
                    model.add(BatchNormalization())

                model.add(Activation('relu'))

                if dropout > 0:
                    model.add(Dropout(rate=dropout))

            model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))

            if optimizer_name == "Adam":
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate_value,
                    name=optimizer_name)

            elif optimizer_name == "Adagrad":
                optimizer = tf.keras.optimizers.Adagrad(
                    learning_rate=learning_rate_value, initial_accumulator_value=0.1, epsilon=1e-07,
                    name='Adagrad'
                )

            elif optimizer_name == "Adamax":
                optimizer = tf.keras.optimizers.Adamax(
                    learning_rate=learning_rate_value, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                    name='Adamax'
                )
            elif optimizer_name == "Adadelta":
                optimizer = tf.keras.optimizers.Adadelta(
                    learning_rate=learning_rate_value, rho=0.95, epsilon=1e-07, name='Adadelta'
                )

            # Compile model
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

            return model

        KerasModel(dense_builder)
