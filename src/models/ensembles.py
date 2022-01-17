from abc import ABC, abstractmethod
from typing import List, Iterable

import numpy as np

from Datasets.Datasets import Dataset
from evaluator.Evaluator import Evaluator
from metrics.Metrics import Metric
from models.Models import Model


class Ensemble(ABC):

    def __init__(self, models: List[Model]):
        self.models = models

    def fit(self, dataset: Dataset):
        for model in self.models:
            model.fit(dataset)

    @abstractmethod
    def predict(self, dataset: Dataset):
        raise NotImplementedError()

    def evaluate(self,
                 dataset: Dataset,
                 metrics: List[Metric],
                 per_task_metrics: bool = False,
                 n_classes: int = 2):
        """
        Evaluates the performance of this model on specified dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        metrics: Metric / List[Metric]
            The set of metrics provided.
        per_task_metrics: bool, optional (default False)
            If true, return computed metric for each task on multitask dataset.
        n_classes: int, optional (default None)
            If specified, will use `n_classes` as the number of unique classes.

        Returns
        -------
        multitask_scores: dict
            Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict, optional
            If `per_task_metrics == True` is passed as a keyword argument,
            then returns a second dictionary of scores for each task
            separately.
        """

        evaluator = Evaluator(self, dataset)

        return evaluator.compute_model_performance(metrics,
                                                   per_task_metrics=per_task_metrics,
                                                   n_classes=n_classes)


class VotingClassifier(Ensemble):

    def __init__(self, models: List[Model], voting="soft"):
        super().__init__(models)
        self.voting = voting

    def predict(self, dataset: Dataset, proba=False):
        assert len(self.models) > 0

        n_labels = len(np.unique(dataset.y))
        results_from_all_models = np.empty(shape=(len(dataset.mols), n_labels, len(self.models)))

        for i, model in enumerate(self.models):
            model_y_predicted = model.predict(dataset)

            for j in range(len(model_y_predicted)):
                for prediction_i, prediction in enumerate(model_y_predicted[j]):
                    results_from_all_models[j, prediction_i, i] = model_y_predicted[j, prediction_i]

        if proba:
            final_result = np.empty(shape=(len(dataset.mols), n_labels))
        else:
            final_result = np.empty(shape=(len(dataset.mols)))

        if self.voting == "soft":

            for mol_i, mol_predictions in enumerate(results_from_all_models):
                class_predictions = np.apply_along_axis(np.mean, 1, mol_predictions)
                max_prediction = 0
                max_prediction_class = 0
                for i, class_prediction in enumerate(class_predictions):
                    if class_prediction > max_prediction:
                        max_prediction_class = i
                        max_prediction = class_prediction

                if proba:
                    final_result[mol_i] = class_predictions
                else:
                    final_result[mol_i] = max_prediction_class

            return final_result

        # TODO : implement
        elif self.voting == "hard":
            for mol_i, mol_predictions in enumerate(results_from_all_models):
                predictions_counter = {}
                max_prediction = 0
                max_prediction_class = 0
                for i, class_prediction in enumerate(mol_predictions):
                    for j in range(len(mol_predictions)):

                        # TODO: instantiate list of prediction classes (maybe before all of this)
                        # and work over that list

                        if class_prediction > max_prediction:
                            max_prediction_class = i
                            max_prediction = class_prediction

                if proba:
                    pass
                else:
                    final_result[mol_i] = max_prediction_class
