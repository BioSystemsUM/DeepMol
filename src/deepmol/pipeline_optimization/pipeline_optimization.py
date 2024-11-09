import os
import warnings
from typing import Union, List

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import StudyDirection

from deepmol.pipeline import Pipeline
from deepmol.pipeline.ensemble import VotingPipeline
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
    n_pipelines_ensemble : int
        Number of pipelines to be used in the ensemble.
    n_jobs : int
        Number of parallel jobs.

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
    pipelines_ensemble : deepmol.pipeline.ensemble.VotingPipeline
        Pipelines ensemble.


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
                 directions: List[Union[str, StudyDirection]] = None,
                 n_pipelines_ensemble=5,
                 n_jobs=5) -> None:
        """
        Initialize the PipelineOptimization class.
        """
        self._pipelines_ensemble = None
        self.direction = direction
        self.study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                         direction=direction, load_if_exists=load_if_exists, directions=directions)
        self.study.set_user_attr("best_scores", {})
        self.n_pipelines_ensemble = n_pipelines_ensemble
        self.n_jobs = n_jobs

        

    def optimize(self, objective_steps: Union[callable, str], n_trials: int, save_top_n: int = 1,
                 objective: Objective = ObjectiveTrainEval,
                 trial_timeout: int = 86400, **kwargs) -> None:
        """
        Optimize the pipeline.

        Parameters
        ----------
        objective_steps : callable or str
            Objective function. If a string is passed, a preset objective function is used.
        n_trials : int
            Number of trials.
        save_top_n : int
            Number of best pipelines to save.
        objective : deepmol.pipeline_optimization.objective_wrapper.Objective
            Objective class.
        trial_timeout : int
            Timeout for each trial in seconds.
        **kwargs
            Additional arguments to be passed to the objective function.
        """
        if isinstance(objective_steps, str):
            assert objective_steps in ['keras', 'deepchem', 'sklearn', 'all'], \
                'objective_steps must be one of the following: keras, deepchem, sklearn, all'
            objective_steps = _get_preset(objective_steps)
        objective = objective(objective_steps, self.study, self.direction,
                              save_top_n, trial_timeout, **kwargs)
        self.study.optimize(objective, n_trials=n_trials, catch=(TimeoutError,))
        if self.best_value in [np.float_('-inf'), np.float_('inf')]:
            raise ValueError('The best value is -inf or inf. No trials completed successfully.')
        if self.n_pipelines_ensemble > 0:
            if save_top_n < self.n_pipelines_ensemble:
                warnings.warn(f'save_top_n ({save_top_n}) is smaller than n_pipelines_ensemble, '
                              f'producing an ensemble pipeline with {save_top_n} pipelines.')
                self.n_pipelines_ensemble = save_top_n
            self.get_pipelines_ensemble()

    def get_pipelines_ensemble(self):
        """
        Returns the best pipelines ensemble.
        """
        trials_df = self.trials_dataframe()
        ascending = True if self.direction == 'minimize' else False
        best_trials = (trials_df[trials_df['state'] == 'COMPLETE'].
                       sort_values('value', ascending=ascending))
        pipelines = []
        i = 0
        while len(pipelines) < self.n_pipelines_ensemble:
            path_exists = False
            while not path_exists:
                number = best_trials.iloc[i]['number']
                path = os.path.join(self.study.study_name, f'trial_{number}')
                if os.path.exists(path):
                    pipelines.append(Pipeline.load(path))
                    path_exists = True
                i += 1

        voting_pipeline = VotingPipeline(pipelines=pipelines, voting='soft')
        voting_pipeline.save(os.path.join(self.study.study_name, 'voting_pipeline'))
        self._pipelines_ensemble = voting_pipeline

    @property
    def pipelines_ensemble(self):
        """
        Returns the pipelines ensemble.
        """
        if self._pipelines_ensemble is None:
            raise AttributeError('pipelines_ensemble attribute is not available. '
                                 'Set a number to pipeline.n_pipelines_ensemble'
                                 'and run pipeline.get_pipelines_ensemble() to get the pipelines ensemble. ')
        return self._pipelines_ensemble

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
