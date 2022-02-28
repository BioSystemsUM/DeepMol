from models.models import Model
from models.sklearn_models import SklearnModel
from metrics.metrics import Metric
from splitters.splitters import RandomSplitter, SingletaskStratifiedSplitter
from typing import Optional, Callable, Sequence
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import numpy as np
from datasets.datasets import Dataset
from sklearn.base import clone


# Only for sequential single input models
class KerasModel(Model):
    """Wrapper class that wraps keras models.
    The `KerasModel` class provides a wrapper around keras
    models that allows this models to be trained on `Dataset` objects.
    """

    def __init__(self,
                 model_builder: Callable,
                 mode: Optional[str] = 'classification',
                 model_dir: Optional[str] = None,
                 loss: Optional[str] = 'binary_crossentropy',
                 optimizer: Optional[str] = 'adam',
                 learning_rate: Optional[float] = 0.001,
                 epochs: Optional[int] = 150,
                 batch_size: Optional[int] = 10,
                 verbose: Optional[int] = 0,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        ...
        """
        super().__init__(model_builder, model_dir, **kwargs)
        '''
        super(KerasModel, self).__init__(model_builder,
                                         model_dir,
                                         mode,
                                         epochs,
                                         batch_size,
                                         verbose,
                                         **kwargs)
        '''

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
            raise ValueError('Only classification or regression is accepted.')

    def fit(self, dataset: Dataset) -> None:
        """Fits keras model to data.
        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        """
        features = dataset.X
        y = np.squeeze(dataset.y)
        self.model.fit(features, y)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """Makes predictions on dataset.
        Parameters
        ----------
        dataset: Dataset
          Dataset to make prediction on.
        Returns
        -------
        np.ndarray
          The value is a return value of `predict_proba` or `predict` method
          of the scikit-learn model. If the scikit-learn model has both methods,
          the value is always a return value of `predict_proba`.
        """
        try:
            return self.model.predict_proba(dataset.X)
        except AttributeError:
            print(self.model)
            print(type(self.model))
            return self.model.predict(dataset.X)

    def predict_on_batch(self, X: Dataset) -> np.ndarray:
        """Makes predictions on batch of data.
        Parameters
        ----------
        X: Dataset
          Dataset to make prediction on.
        """
        return super(KerasModel, self).predict(X)

    # TODO: functions from sklearnModels, adapt to kerasModels if needed
    '''
    def save(self):
        """Saves scikit-learn model to disk using joblib."""
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """Loads scikit-learn model from joblib file on disk."""
        self.model = load_from_disk(self.get_model_filename(self.model_dir))
    '''

    def fit_on_batch(self, X: Sequence, y: Sequence):
        pass

    def reload(self) -> None:
        pass

    def save(self) -> None:
        pass

    def get_task_type(self) -> str:
        pass

    def get_num_tasks(self) -> int:
        pass

    def cross_validate(self,
                       dataset: Dataset,
                       metric: Metric,
                       folds: int = 3):

        # TODO: add option to choose between splitters
        splitter = None
        if self.mode == 'classification':
            splitter = SingletaskStratifiedSplitter()
        if self.mode == 'regression':
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

            print('Train Score: ')
            train_score = dummy_model.evaluate(train_ds, metric)
            train_scores.append(train_score[metric.name])
            avg_train_score += train_score[metric.name]

            print('Test Score: ')
            test_score = dummy_model.evaluate(test_ds, metric)
            test_scores.append(test_score[metric.name])
            avg_test_score += test_score[metric.name]

            if test_score[metric.name] > test_score_best_model:
                test_score_best_model = test_score[metric.name]
                train_score_best_model = train_score[metric.name]
                best_model = dummy_model

        return best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, \
               avg_train_score / folds, avg_test_score / folds
