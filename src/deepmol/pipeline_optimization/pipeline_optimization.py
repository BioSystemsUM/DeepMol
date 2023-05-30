from typing import Union

import optuna

from deepmol.datasets import Dataset
from deepmol.pipeline_optimization._utils import _get_preset


class PipelineOptimization:

    def __init__(self, direction: str):
        self.direction = direction
        self.study = optuna.create_study(direction=direction)

    def optimize(self, dataset: Dataset, objective: Union[callable, str], n_trials: int):
        if isinstance(objective, str):
            objective = _get_preset(dataset, objective)
        self.study.optimize(objective, n_trials=n_trials)

    def best_params(self):
        return self.study.best_params

    def best_trial(self):
        return self.study.best_trial

    def best_value(self):
        return self.study.best_value
