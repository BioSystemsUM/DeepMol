import shutil
from unittest import TestCase

from sklearn.metrics import roc_auc_score, precision_score, classification_report, accuracy_score, confusion_matrix

from deepmol.metrics import Metric
from deepmol.models import KerasModel
from unit_tests.models.test_models import ModelsTestCase

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise, Conv1D, Flatten, Reshape
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop


def make_cnn_model(input_dim=None,
                   g_noise=0.05,
                   DENSE=128,
                   DROPOUT=0.5,
                   C1_K=8,
                   C1_S=32,
                   C2_K=16,
                   C2_S=32,
                   activation='relu',
                   loss='binary_crossentropy',
                   optimizer='adadelta',
                   learning_rate=0.01,
                   metrics='accuracy'):
    model = Sequential()
    # Adding a bit of GaussianNoise also works as regularization
    model.add(GaussianNoise(g_noise, input_shape=(input_dim,)))
    # First two is number of filter + kernel size
    model.add(Reshape((input_dim, 1)))
    model.add(Conv1D(C1_K, (C1_S), activation=activation, padding="same"))
    model.add(Conv1D(C2_K, (C2_S), padding="same", activation=activation))
    model.add(Flatten())
    model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer == 'adadelta':
        opt = Adadelta(lr=learning_rate)
    elif optimizer == 'adam':
        opt = Adam(lr=learning_rate)
    elif optimizer == 'rsmprop':
        opt = RMSprop(lr=learning_rate)
    else:
        opt = optimizer

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


class TestKerasModel(ModelsTestCase, TestCase):

    def test_fit_predict_evaluate(self):
        model = KerasModel(model_builder=make_cnn_model,
                           epochs=2, input_dim=self.binary_dataset.X.shape[1])
        model.fit(self.binary_dataset)

        test_preds = model.predict(self.binary_dataset_test)
        self.assertEqual(len(test_preds), len(self.binary_dataset_test))
        # evaluate the model
        for pred in test_preds:
            self.assertAlmostEqual(sum(pred), 1, delta=0.0001)

        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        evaluate = model.evaluate(self.binary_dataset_test, metrics)
        self.assertEqual(len(evaluate[0]), len(metrics))
        self.assertEqual(evaluate[1], {})
        self.assertTrue('roc_auc_score' in evaluate[0].keys())
        self.assertTrue('precision_score' in evaluate[0].keys())
        self.assertTrue('accuracy_score' in evaluate[0].keys())
        self.assertTrue('confusion_matrix' in evaluate[0].keys())
        self.assertTrue('classification_report' in evaluate[0].keys())

    def test_save_model(self):
        model = KerasModel(model_builder=make_cnn_model,
                           epochs=2, input_dim=self.binary_dataset.X.shape[1])
        model.fit(self.binary_dataset)

        first_predictions = model.predict(self.binary_dataset_test)

        model.save("test_model")
        loaded_model = KerasModel.load("test_model")
        loaded_model_predictions = loaded_model.predict(self.binary_dataset_test)
        for i in range(len(first_predictions)):
            self.assertEqual(first_predictions[i][0], loaded_model_predictions[i][0])

        shutil.rmtree("test_model")