from unittest import TestCase

from sklearn.metrics import accuracy_score

from deepmol.metrics import Metric
from deepmol.parameter_optimization import HyperparameterOptimizerValidation
from unit_tests.models.test_models import ModelsTestCase
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_dim, optimizer='adam', dropout=0.5):
    # create model
    inputs = layers.Input(shape=input_dim)

    # Define the shared layers
    shared_layer_1 = layers.Dense(64, activation="relu")
    dropout_1 = Dropout(dropout)
    shared_layer_2 = layers.Dense(32, activation="relu")

    # Define the shared layers for the inputs
    x = shared_layer_1(inputs)
    x = dropout_1(x)
    x = shared_layer_2(x)

    task_output = layers.Dense(1, activation="sigmoid")(x)

    # Define the model that outputs the predictions for each task
    model = keras.Model(inputs=inputs, outputs=task_output)
    # Compile the model with different loss functions and metrics for each task
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


class TestKerasHyperparameterOptimization(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        optimizer = HyperparameterOptimizerValidation(create_model)
        params_dict_dense = {
            "input_dim": [self.binary_dataset.X.shape[1]],
            "dropout": [0.5, 0.6, 0.7],
            "optimizer": ['adam']
        }

        best_dnn, best_hyperparams, all_results = optimizer.hyperparameter_search(train_dataset=self.binary_dataset,
                                                                                  valid_dataset=self.binary_dataset,
                                                                                  metric=Metric(accuracy_score),
                                                                                  maximize_metric=True,
                                                                                  n_iter_search=2,
                                                                                  params_dict=params_dict_dense,
                                                                                  model_type="keras"
                                                                                  )
