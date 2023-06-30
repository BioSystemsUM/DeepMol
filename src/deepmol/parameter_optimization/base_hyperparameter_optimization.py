from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from deepmol.datasets import Dataset
from deepmol.loggers import Logger
from deepmol.metrics import Metric
from deepmol.models import Model


class HyperparameterOptimizer(ABC):
    """
    Abstract superclass for hyperparameter search classes.
    """

    def __init__(self, model_builder: callable,
                 model_type: str,
                 params_dict: Dict,
                 metric: Metric,
                 maximize_metric: bool,
                 n_iter_search: int = 15,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 logdir: str = None,
                 mode: str = None,
                 **kwargs):
        """
        Initialize Hyperparameter Optimizer.
        Note this is an abstract constructor which should only be used by subclasses.

        Parameters
        ----------
        model_builder: callable
            This parameter must be constructor function which returns an object which is an instance of `Models`.
            This function must accept two arguments, `model_params` of type `dict` and 'model_dir', a string specifying
            a path to a model directory.
        model_type: str
            The type of model to use. Can be 'keras' or 'sklearn'.
        params_dict: Dict
            Dictionary mapping hyperparameter names (strings) to lists of possible parameter values.
        metric: Metric
            The metric to optimize.
        maximize_metric: bool
            If True, return the model with the highest score.
        n_iter_search: int
            Number of random combinations of parameters to test, if None performs complete grid search.
        n_jobs: int
            Number of jobs to run in parallel.
        verbose: int
            Controls the verbosity: the higher, the more messages.
        logdir: str
            The directory in which to store created models. If not set, will use a temporary directory.
        mode: str
            The mode of the model. Can be 'classification' or 'regression'.
        """
        if self.__class__.__name__ == "HyperparamOpt":
            raise ValueError("HyperparamOpt is an abstract superclass and cannot be directly instantiated. "
                             "You probably want to instantiate a concrete subclass instead.")
        self.model_builder = model_builder
        self.mode = mode
        self.model_type = model_type
        self.params_dict = params_dict
        self.metric = metric
        self.maximize_metric = maximize_metric
        self.n_iter_search = n_iter_search
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.logdir = logdir
        self.kwargs = kwargs
        self.logger = Logger()

    @abstractmethod
    def fit(self, train_dataset: Dataset, valid_dataset: Dataset = None) \
            -> Tuple[Model, Dict[str, Any], Dict[str, float]]:
        """
        Conduct Hyperparameter search.

        This method defines the common API shared by all hyperparameter optimization subclasses. Different classes will
        implement different search methods, but they must all follow this common API.

        Parameters
        ----------
        train_dataset: Dataset
            The training dataset.
        valid_dataset: Dataset
            The validation dataset.
        Returns
        -------
        Tuple[Model, Dict[str, Any], Dict[str, float]]:
            A tuple containing the best model, the best hyperparameters, and all scores.
        """