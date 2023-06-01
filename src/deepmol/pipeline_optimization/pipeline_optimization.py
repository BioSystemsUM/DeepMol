from typing import Union, List

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import StudyDirection

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline_optimization._utils import _get_preset


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

    def optimize(self, train_dataset: Dataset, test_dataset: Dataset, objective: Union[callable, str], metric: Metric,
                 n_trials: int):
        if isinstance(objective, str):
            objective = _get_preset(objective)
        self.study.optimize(lambda trial: objective(trial, train_dataset, test_dataset, metric), n_trials=n_trials)

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
