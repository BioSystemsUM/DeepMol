import os
import shutil
import tempfile
from abc import ABC
from typing import List, Union, Tuple, Dict
import numpy as np

from deepmol.base import Predictor
from deepmol.datasets import Dataset
from deepmol.evaluator.evaluator import Evaluator
from deepmol.loggers.logger import Logger
from deepmol.metrics.metrics import Metric

from sklearn.base import BaseEstimator

from deepmol.models._utils import _return_invalid


class Model(BaseEstimator, Predictor, ABC):
    """
    Abstract base class for ML/DL models.
    """

    def __init__(self, model: BaseEstimator = None, model_dir: str = None, **kwargs) -> None:
        """
        Abstract class for all models.
        This is an abstact class and should not be invoked directly.

        Parameters
        ----------
        model: BaseEstimator
            Wrapper around ScikitLearn/Keras/Tensorflow/DeepChem model object.
        model_dir: str
            Path to directory where model will be stored. If not specified, model will be stored in a temporary
            directory.
        """
        if self.__class__.__name__ == "Model":
            raise ValueError(
                "This constructor is for an abstract class and should never be called directly. Can only call from "
                "subclass constructors.")

        super().__init__()

        self.model_dir_is_temp = False

        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        else:
            model_dir = tempfile.mkdtemp()
            self.model_dir_is_temp = True

        self._model_dir = model_dir
        self.model = model
        self.model_class = model.__class__

        self.logger = Logger()

    def __del__(self):
        """
        Delete model directory if it was created by this object.
        """
        if 'model_dir_is_temp' in dir(self) and self.model_dir_is_temp:
            shutil.rmtree(self.model_dir)

    def fit_on_batch(self, dataset: Dataset) -> None:
        """
        Perform a single step of training.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """

    def predict_on_batch(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on given batch of new data.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
    @classmethod
    def load(cls, folder_path: str) -> 'Model':
        """
        Reload trained model from disk.

        Parameters
        ----------
        folder_path: str
            Path to folder where model is stored.

        Returns
        -------
        Model
            Model object.
        """

    @staticmethod
    def get_model_filename(model_dir: str) -> str:
        """
        Given model directory, obtain filename for the model itself.

        Parameters
        ----------
        model_dir: str
            Path to directory where model is stored.

        Returns
        -------
        str
            Path to model file.
        """
        return os.path.join(model_dir, "model.pkl")

    @staticmethod
    def get_params_filename(model_dir: str) -> str:
        """
        Given model directory, obtain filename for the model itself.

        Parameters
        ----------
        model_dir: str
            Path to directory where model is stored.

        Returns
        -------
        str
            Path to file where model parameters are stored.
        """
        return os.path.join(model_dir, "model_params.joblib")

    def save(self, file_path: str = None) -> None:
        """
        Function for saving models.
        Each subclass is responsible for overriding this method.

        Parameters
        ----------
        file_path: str
            Path to file where model should be saved.
        """

    def predict(self, dataset: Dataset, return_invalid: bool = False) -> np.ndarray:
        """
        Uses self to make predictions on provided Dataset object.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on
        
        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        np.ndarray
            A numpy array of predictions.
        """
        y_preds = []
        for (X_batch, _, _, ids_batch) in dataset.iterbatches(deterministic=True):
            n_samples = len(X_batch)
            y_pred_batch = self.predict_on_batch(X_batch)
            # Discard any padded predictions
            y_pred_batch = y_pred_batch[:n_samples]
            y_preds.append(y_pred_batch)
        y_pred = np.concatenate(y_preds)

        if return_invalid:
            y_pred = _return_invalid(dataset, y_pred)

        return y_pred

    def predict_proba(self, dataset: Dataset, return_invalid: bool = False) -> np.ndarray:
        """
        Uses self to make predictions on provided Dataset object.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on

        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        np.ndarray
            A numpy array of predictions.
        """
        y_pred = self.model.predict_proba(dataset.X)

        if return_invalid:
            y_pred = _return_invalid(dataset, y_pred)
        return y_pred

    def evaluate(self,
                 dataset: Dataset,
                 metrics: Union[List[Metric], Metric],
                 per_task_metrics: bool = False) -> Tuple[Dict, Union[None, Dict]]:
        """
        Evaluates the performance of this model on specified dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        metrics: Union[List[Metric], Metric]
            The set of metrics provided.
        per_task_metrics: bool
            If true, return computed metric for each task on multitask dataset.
        kwargs:
            Additional keyword arguments to pass to `Evaluator.compute_model_performance`.

        Returns
        -------
        multitask_scores: dict
            Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict, optional
            If `per_task_metrics == True` is passed as a keyword argument, then returns a second dictionary of scores
            for each task separately.
        """
        evaluator = Evaluator(self, dataset)
        return evaluator.compute_model_performance(metrics, per_task_metrics=per_task_metrics)

    def get_task_type(self) -> str:
        """
        Currently models can only be classifiers or regressors.
        """

    def get_num_tasks(self) -> int:
        """
        Get number of tasks.
        """
