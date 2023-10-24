import os
from typing import Union, List

import optuna
import pandas as pd
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import StudyDirection

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization._utils import _get_preset
from deepmol.pipeline_optimization.objective_wrapper import ObjectiveTrainEval, Objective


class PipelineOptimization:
    """
    Class for optimizing a pipeline with Optuna.
    It can optimize all steps of a pipeline and respective hyperparameters.

    Parameters
    ----------
    storage : str or optuna.storages.BaseStorage
        Database storage URL such as sqlite:///example.db.
        If None, in-memory storage is used.
    sampler : optuna.samplers.BaseSampler
        A sampler object that implements background algorithm for value suggestion.
        If None, optuna.samplers.TPESampler is used as the default.
    pruner : optuna.pruners.BasePruner
        A pruner object that decides early stopping of unpromising trials.
        If None, optuna.pruners.MedianPruner is used as the default.
    study_name : str
        Study's name. If this argument is set to None, a unique name is generated automatically.
    direction : str or optuna.study.StudyDirection
        Direction of the optimization (minimize or maximize).
    load_if_exists : bool
        Flag to control the behavior to handle a conflict of study names.
        If set to True, the study will be loaded instead of raising an exception.
    directions : list of str or optuna.study.StudyDirection
        Direction of the optimization for each step of the pipeline.
        If None, the direction argument is used for all steps.

    Attributes
    ----------
    best_params : dict
        Dictionary with the best hyperparameters.
    best_trial : optuna.trial.FrozenTrial
        Best trial.
    best_value : float
        Best value.
    trials : list of optuna.trial.FrozenTrial
        List of all trials.
    best_pipeline : deepmol.pipeline.Pipeline
        Best pipeline.

    Examples
    --------
    >>> from deepmol.loaders import CSVLoader
    >>> from deepmol.pipeline_optimization import PipelineOptimization
    >>> from deepmol.metrics import Metric
    >>> from deepmol.splitters import RandomSplitter
    >>> from sklearn.metrics import accuracy_score
    >>> dataset_path = "dataset.csv"
    >>> loader = CSVLoader(dataset_path=dataset_path,
    >>>                    smiles_field='Smiles',
    >>>                    labels_fields=['Class'])
    >>> dataset_smiles = loader.create_dataset(sep=";")
    >>> po = PipelineOptimization(direction='maximize', study_name='test_pipeline')
    >>> metric = Metric(accuracy_score)
    >>> train, test = RandomSplitter().train_test_split(dataset_smiles, seed=123)
    >>> po.optimize(train_dataset=train, test_dataset=test, objective_steps='classification_objective', metric=metric,
    >>>             n_trials=3, data=train, save_top_n=1)
    """

    def __init__(self,
                 storage: Union[str, BaseStorage] = None,
                 sampler: BaseSampler = None,
                 pruner: BasePruner = None,
                 study_name: str = None,
                 direction: Union[str, StudyDirection] = None,
                 load_if_exists: bool = False,
                 directions: List[Union[str, StudyDirection]] = None) -> None:
        """
        Initialize the PipelineOptimization class.
        """
        self.direction = direction
        self.study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                         direction=direction, load_if_exists=load_if_exists, directions=directions)
        self.study.set_user_attr("best_scores", {})

    def optimize(self, train_dataset: Dataset, test_dataset: Dataset, objective_steps: Union[callable, str],
                 metric: Metric, n_trials: int, save_top_n: int = 1, objective: Objective = ObjectiveTrainEval,
                 **kwargs) -> None:
        """
        Optimize the pipeline.

        Parameters
        ----------
        train_dataset : deepmol.datasets.Dataset
            Training dataset.
        test_dataset : deepmol.datasets.Dataset
            Test dataset.
        objective_steps : callable or str
            Objective function. If a string is passed, a preset objective function is used.
        metric : deepmol.metrics.Metric
            Metric to be used.
        n_trials : int
            Number of trials.
        save_top_n : int
            Number of best pipelines to save.
        objective : deepmol.pipeline_optimization.objective_wrapper.Objective
            Objective class.
        **kwargs
            Additional arguments to be passed to the objective function.
        """
        if isinstance(objective_steps, str):
            assert objective_steps in ['keras', 'deepchem', 'sklearn', 'all'], \
                'objective_steps must be one of the following: keras, deepchem, sklearn, all'
            objective_steps = _get_preset(objective_steps)
        objective = objective(objective_steps, self.study, self.direction, train_dataset, test_dataset, metric,
                                       save_top_n, **kwargs)
        self.study.optimize(objective, n_trials=n_trials)

    @property
    def best_params(self):
        """
        Returns the best hyperparameters.
        """
        return self.study.best_params

    @property
    def best_trial(self):
        """
        Returns the best trial.
        """
        return self.study.best_trial

    @property
    def best_value(self):
        """
        Returns the best value (score of the best trial).
        """
        return self.study.best_value

    @property
    def trials(self):
        """
        Returns all trials.
        """
        return self.study.trials

    @property
    def best_pipeline(self):
        """
        Returns the best pipeline.
        """
        best_trial_id = self.study.best_trial.number
        path = os.path.join(self.study.study_name, f'trial_{best_trial_id}')
        return Pipeline.load(path)

    def trials_dataframe(self, cols: List[str] = None) -> pd.DataFrame:
        """
        Returns the trials dataframe.

        Parameters
        ----------
        cols : list of str
            Columns to be returned.

        Returns
        -------
        pd.DataFrame
            Trials dataframe.
        """
        if cols is not None:
            return self.study.trials_dataframe(attrs=tuple(cols))
        return self.study.trials_dataframe()

    def get_param_importances(self):
        """
        Returns the parameter importances.
        """
        try:
            return optuna.importance.get_param_importances(self.study)
        except RuntimeError:
            return None
