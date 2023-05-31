from typing import Union

import optuna

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.pipeline_optimization._utils import _get_preset


class PipelineOptimization:

    def __init__(self, direction: str):
        self.direction = direction
        self.study = optuna.create_study(direction=direction)

    def optimize(self, train_dataset: Dataset, test_dataset: Dataset, objective: Union[callable, str], metric: Metric, n_trials: int):
        if isinstance(objective, str):
            objective = _get_preset(train_dataset, objective)
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
