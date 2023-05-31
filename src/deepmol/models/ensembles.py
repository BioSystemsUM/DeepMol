from abc import ABC, abstractmethod
from typing import List

import numpy as np

from deepmol.base import Predictor
from deepmol.datasets import Dataset
from deepmol.evaluator.evaluator import Evaluator
from deepmol.metrics.metrics import Metric
from deepmol.models.models import Model


class Ensemble(ABC, Predictor):
    """
    Abstract class for ensembles of models.
    """

    def __init__(self, models: List[Model]):
        """
        Initializes an ensemble of models.

        Parameters
        ----------
        models: List[Model]
            List of models to be used in the ensemble.
        """
        super().__init__()
        self.models = models

    def fit(self, dataset: Dataset):
        """
        Fits the models to the specified dataset.
        """
        for model in self.models:
            model.fit(dataset)

    @abstractmethod
    def predict(self, dataset: Dataset):
        """
        Predicts the labels for the specified dataset.
        """

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
        metrics: List[Metric]
            The set of metrics provided.
        per_task_metrics: bool
            If true, return computed metric for each task on multitask dataset.
        n_classes: int
            If specified, will use `n_classes` as the number of unique classes.

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


class VotingClassifier(Ensemble):
    """
    VotingClassifier Ensemble.
    It uses a voting strategy to predict the labels of a dataset.
    """

    def __init__(self, models: List[Model], voting: str = "soft"):
        """
        Initializes a VotingClassifier ensemble.

        Parameters
        ----------
        models: List[Model]
            List of models to be used in the ensemble.
        voting: str
            Voting strategy to use. Can be either 'soft' or 'hard'.
        """
        super().__init__(models)
        self.voting = voting

    def predict(self, dataset: Dataset, proba: bool = False):
        """
        Predicts the labels for the specified dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        proba: bool
            If true, returns the probabilities instead of class labels.

        Returns
        -------
        final_result: np.ndarray
            Predicted labels or probabilities.
        """
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

                if proba:
                    final_result[mol_i] = class_predictions
                else:
                    max_prediction = 0
                    max_prediction_class = 0
                    for i, class_prediction in enumerate(class_predictions):
                        if class_prediction > max_prediction:
                            max_prediction_class = i
                            max_prediction = class_prediction
                    final_result[mol_i] = max_prediction_class

            return final_result

        elif self.voting == "hard":
            for mol_i, mol_predictions in enumerate(results_from_all_models):
                predictions_counter = {}
                for i, models_class_predictions in enumerate(mol_predictions):
                    for model_class_prediction in models_class_predictions:
                        if model_class_prediction > 0.5:

                            if i in predictions_counter:
                                predictions_counter[i].append(model_class_prediction)
                            else:
                                predictions_counter[i] = [model_class_prediction]

                class_with_more_predictions = None
                max_n_predictions = 0

                for class_ in predictions_counter:
                    len_predictions_counter = len(predictions_counter[class_])
                    if len_predictions_counter > max_n_predictions:
                        max_n_predictions = len_predictions_counter
                        class_with_more_predictions = class_

                assert class_with_more_predictions is not None

                final_result[mol_i] = class_with_more_predictions

        else:
            raise Exception("Voting has to be either 'soft' or 'hard'")

        return final_result
