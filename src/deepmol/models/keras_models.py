from deepmol.models.models import Model
from deepmol.models.sklearn_models import SklearnModel
from deepmol.metrics.metrics import Metric
from deepmol.splitters.splitters import RandomSplitter, SingletaskStratifiedSplitter
from typing import Sequence
import numpy as np
from deepmol.datasets import Dataset
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.base import clone


# Only for sequential single input models
class KerasModel(Model):
    """
    Wrapper class that wraps keras models.
    The `KerasModel` class provides a wrapper around keras models that allows this models to be trained on `Dataset`
    objects.
    """

    def __init__(self,
                 model_builder: callable,
                 mode: str = 'classification',
                 model_dir: str = None,
                 loss: str = 'binary_crossentropy',
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
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
        mode: str
            The mode of the model. Can be either 'classification' or 'regression'.
        model_dir: str
            The directory to save the model to.
        loss: str
            The loss function to use.
        optimizer: str
            The optimizer to use.
        learning_rate: float
            The learning rate to use.
        epochs: int
            The number of epochs to train for.
        batch_size: int
            The batch size to use.
        verbose: int
            The verbosity of the model.
        """
        super().__init__(model_builder, model_dir, **kwargs)
        self.mode = mode
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model_type = 'keras'
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_builder = model_builder
        self.verbose = verbose

        if mode == 'classification':
            self.model = KerasClassifier(build_fn=model_builder, epochs=epochs, batch_size=batch_size,
                                         verbose=verbose, **kwargs)
        elif mode == 'regression':
            self.model = KerasRegressor(build_fn=model_builder, nb_epoch=epochs, batch_size=batch_size, verbose=verbose,
                                        **kwargs)
        else:
            self.model = model_builder

    def fit(self, dataset: Dataset, **kwargs) -> None:
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
        try:
            return self.model.predict_proba(dataset.X.astype('float32'))
        except AttributeError:
            self.logger.info(str(self.model))
            self.logger.info(str(type(self.model)))
            return self.model.predict(dataset.X.astype('float32'))

    def predict_on_batch(self, X: Dataset) -> np.ndarray:
        """
        Makes predictions on batch of data.

        Parameters
        ----------
        X: Dataset
          Dataset to make prediction on.

        Returns
        -------
        np.ndarray
            numpy array of predictions.
        """
        return super(KerasModel, self).predict(X)

    def fit_on_batch(self, X: Sequence, y: Sequence):
        """
        Fits model on batch of data.
        """

    def reload(self) -> None:
        """
        Reloads the model from disk.
        """

    def save(self) -> None:
        """
        Saves the model to disk.
        """

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
                       folds: int = 3):
        """
        Cross validates the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to cross validate on.
        metric: Metric
            The metric to use for cross validation.
        folds: int
            The number of folds to use for cross validation.

        Returns
        -------
        Tuple[SKlearnModel, float, float, List[float], List[float], float, float]
            The first element is the best model, the second is the train score of the best model, the third is the train
            score of the best model, the fourth is the test scores of all models, the fifth is the average train scores
            of all folds and the sixth is the average test score of all folds.
        """
        # TODO: add option to choose between splitters
        splitter = None
        if dataset.mode == 'classification':
            splitter = SingletaskStratifiedSplitter()
        if dataset.mode == 'regression':
            splitter = RandomSplitter()

        assert splitter is not None

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
