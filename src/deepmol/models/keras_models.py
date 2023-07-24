import os
from typing import Union

import keras

from deepmol.models._utils import _get_splitter, load_from_disk, _save_keras_model, get_prediction_from_proba
from deepmol.models.models import Model
from deepmol.models.sklearn_models import SklearnModel
from deepmol.metrics.metrics import Metric
from deepmol.splitters.splitters import Splitter
import numpy as np
from deepmol.datasets import Dataset
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.base import clone

from deepmol.utils.utils import normalize_labels_shape


# Only for sequential single input models
class KerasModel(Model):
    """
    Wrapper class that wraps keras models.
    The `KerasModel` class provides a wrapper around keras models that allows this models to be trained on `Dataset`
    objects.
    """

    def __init__(self,
                 model_builder: callable,
                 mode: Union[str, list] = 'classification',
                 model_dir: str = None,
                 epochs: int = 150,
                 batch_size: int = 10,
                 verbose: int = 0,
                 **kwargs) -> None:
        """
        Initializes a `KerasModel` object.

        Parameters
        ----------
        model_builder: callable
            A function that builds a keras model.
        mode: Union[str, list]
            The mode of the model. Can be either 'classification' or 'regression'.
        model_dir: str
            The directory to save the model to.
        epochs: int
            The number of epochs to train for.
        batch_size: int
            The batch size to use.
        verbose: int
            The verbosity of the model.
        """
        self.mode = mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_builder = model_builder
        self.verbose = verbose

        self.parameters_to_save = {'mode': self.mode,
                                   'batch_size': self.batch_size,
                                   'epochs': self.epochs,
                                   'verbose': self.verbose,
                                   **kwargs}

        if mode == 'classification':
            self.model = KerasClassifier(build_fn=model_builder, epochs=epochs, batch_size=batch_size,
                                         verbose=verbose, **kwargs)
        elif mode == 'regression':
            self.model = KerasRegressor(build_fn=model_builder, nb_epoch=epochs, batch_size=batch_size, verbose=verbose,
                                        **kwargs)
        elif isinstance(model_builder, keras.models.Model):
            self.model = model_builder

        else:
            self.model = model_builder(**kwargs)

        super().__init__(self.model, model_dir, **kwargs)

    @property
    def model_type(self):
        """
        Returns the type of the model.
        """
        return 'keras'

    def _fit(self, dataset: Dataset, **kwargs) -> None:
        """
        Fits keras model to data.

        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        kwargs:
            Additional arguments to pass to `fit` method of the keras model.
        """
        if self.mode != dataset.mode:
            raise ValueError('Dataset mode does not match model mode.')

        features = dataset.X.astype('float32')
        if len(dataset.label_names) == 1:
            y = np.squeeze(dataset.y)
        else:
            targets = [dataset.y[:, i] for i in range(len(dataset.label_names))]
            y = {f"{dataset.label_names[i]}": targets[i] for i in range(len(dataset.label_names))}
        self.model.fit(features, y, **kwargs)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on dataset.

        Parameters
        ----------
        dataset: Dataset
          Dataset to make prediction on.

        Returns
        -------
        np.ndarray
          The value is a return value of `predict_proba` or `predict` method of the scikit-learn model. If the
          scikit-learn model has both methods, the value is always a return value of `predict_proba`.
        """
        predictions = self.predict_proba(dataset)
        y_pred_rounded = get_prediction_from_proba(dataset, predictions)
        return y_pred_rounded

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on.

        Returns
        -------
        np.ndarray
            predictions
        """
        try:
            predictions = self.model.predict_proba(dataset.X.astype('float32'))
        except AttributeError as e:
            self.logger.error(e)
            self.logger.info(str(self.model))
            self.logger.info(str(type(self.model)))
            predictions = self.model.predict(dataset.X.astype('float32'))

        predictions = np.array(predictions)
        if predictions.shape != (len(dataset.mols), dataset.n_tasks):
            predictions = normalize_labels_shape(predictions, dataset.n_tasks)

        if len(predictions.shape) > 1:
            if predictions.shape[1] == len(dataset.mols) and predictions.shape[0] == dataset.n_tasks:
                predictions = predictions.T

        return predictions

    def predict_on_batch(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on batch of data.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on.

        Returns
        -------
        np.ndarray
            numpy array of predictions.
        """
        return super(KerasModel, self).predict(dataset)

    def fit_on_batch(self, dataset: Dataset) -> None:
        """
        Fits model on batch of data.

        Parameters
        ----------
        dataset: Dataset
            Dataset to fit model on.
        """

    @classmethod
    def load(cls, folder_path: str) -> 'KerasModel':
        """
        Reloads the model from disk.

        Parameters
        ----------
        folder_path: str
            The folder path to load the model from.

        Returns
        -------
        KerasModel
            The loaded model.
        """
        file_path_model_builder = os.path.join(folder_path, 'model_builder.pkl')
        model_builder = load_from_disk(file_path_model_builder)
        file_path_model = os.path.join(folder_path, 'model.h5')
        model = keras.models.load_model(file_path_model)
        model_parameters = load_from_disk(os.path.join(folder_path, 'model_parameters.pkl'))
        keras_model_class = cls(model_builder=model_builder, **model_parameters)
        if isinstance(keras_model_class.model, KerasClassifier) or isinstance(keras_model_class.model,
                                                                              KerasRegressor):
            keras_model_class.model.model = model
        else:
            keras_model_class.model = model
        return keras_model_class

    def save(self, file_path: str = None) -> None:
        """
        Saves the model to disk.

        Parameters
        ----------
        file_path: str
            The path to save the model to.
        """
        if file_path is None:
            if self.model_dir is None:
                raise ValueError('No model directory specified.')
            else:
                try:
                    # write self in pickle format
                    _save_keras_model(self.model_dir, self.model.model, self.parameters_to_save, self.model_builder)
                except AttributeError:
                    # write self in pickle format
                    _save_keras_model(self.model_dir, self.model, self.parameters_to_save, self.model_builder)
        else:
            try:
                # write self in pickle format
                _save_keras_model(file_path, self.model.model, self.parameters_to_save, self.model_builder)
            except AttributeError:
                # write self in pickle format
                _save_keras_model(file_path, self.model, self.parameters_to_save, self.model_builder)

    def get_task_type(self) -> str:
        """
        Returns the task type of the model.
        """

    def get_num_tasks(self) -> int:
        """
        Returns the number of tasks of the model.
        """

    def cross_validate(self,
                       dataset: Dataset,
                       metric: Metric,
                       splitter: Splitter = None,
                       folds: int = 3):
        """
        Cross validates the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to cross validate on.
        metric: Metric
            The metric to use for cross validation.
        splitter: Splitter
            The splitter to use for cross validation.
        folds: int
            The number of folds to use for cross validation.

        Returns
        -------
        Tuple[SKlearnModel, float, float, List[float], List[float], float, float]
            The first element is the best model, the second is the train score of the best model, the third is the train
            score of the best model, the fourth is the test scores of all models, the fifth is the average train scores
            of all folds and the sixth is the average test score of all folds.
        """
        if splitter is None:
            splitter = _get_splitter(dataset)

        datasets = splitter.k_fold_split(dataset, folds)

        train_scores = []
        train_score_best_model = 0
        avg_train_score = 0

        test_scores = []
        test_score_best_model = 0
        avg_test_score = 0
        best_model = None
        for train_ds, test_ds in datasets:
            dummy_model = clone(SklearnModel(model=self.model))

            dummy_model.fit(train_ds)

            train_score = dummy_model.evaluate(train_ds, metric)
            train_scores.append(train_score[metric.name])
            avg_train_score += train_score[metric.name]

            test_score = dummy_model.evaluate(test_ds, metric)
            test_scores.append(test_score[metric.name])
            avg_test_score += test_score[metric.name]

            if test_score[metric.name] > test_score_best_model:
                test_score_best_model = test_score[metric.name]
                train_score_best_model = train_score[metric.name]
                best_model = dummy_model

        return best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, avg_train_score / folds, avg_test_score / folds
