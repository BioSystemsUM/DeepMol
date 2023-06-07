import os
from typing import Union, List

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import StudyDirection

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization._utils import _get_preset
from deepmol.pipeline_optimization.objective_wrapper import Objective


class PipelineOptimization:

    def __init__(self,
                 storage: Union[str, BaseStorage] = None,
                 sampler: BaseSampler = None,
                 pruner: BasePruner = None,
                 study_name: str = None,
                 direction: Union[str, StudyDirection] = None,
                 load_if_exists: bool = False,
                 directions: List[Union[str, StudyDirection]] = None) -> None:
        self.direction = direction
        self.study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
                                         direction=direction, load_if_exists=load_if_exists, directions=directions)
        self.study.set_user_attr("best_scores", {})

    def optimize(self, train_dataset: Dataset, test_dataset: Dataset, objective_steps: Union[callable, str],
                 metric: Metric, n_trials: int, save_top_n: int = 1, **kwargs):
        if isinstance(objective_steps, str):
            objective_steps = _get_preset(objective_steps)
        objective = Objective(objective_steps, self.study, self.direction, train_dataset, test_dataset, metric,
                              save_top_n, **kwargs)
        self.study.optimize(objective, n_trials=n_trials)

    @property
    def best_params(self):
        return self.study.best_params

    @property
    def best_trial(self):
        return self.study.best_trial

    @property
    def best_value(self):
        return self.study.best_value

    @property
    def trials(self):
        return self.study.trials

    @property
    def best_pipeline(self):
        best_trial_id = self.study.best_trial.number
        path = os.path.join(self.study.study_name, f'trial_{best_trial_id}')
        return Pipeline.load(path)
