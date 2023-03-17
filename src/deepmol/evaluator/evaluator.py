import csv
from typing import Dict, Union, List, Tuple

import numpy as np

from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.utils.utils import normalize_labels_shape


class Evaluator:
    """
    Class that evaluates a model on a given dataset.
    The evaluator class is used to evaluate a `Model` class on a given `Dataset` object.
    """

    def __init__(self, model: 'Model', dataset: Dataset) -> None:
        """
        Initialize this evaluator.

        Parameters
        ----------
        model: Model
            Model to evaluate. Note that this must be a regression or classification model.
        dataset: Dataset
            Dataset object to evaluate `model` on.
        """
        self.model = model
        self.dataset = dataset

    @staticmethod
    def output_statistics(scores: Dict[str, float], stats_out: str) -> None:
        """
        Write computed stats to file.

        Parameters
        ----------
        scores: Score
            Dictionary mapping names of metrics to scores.
        stats_out: str
            Name of file to write scores to.
        """
        with open(stats_out, "w") as stats_file:
            stats_file.write(str(scores) + "\n")

    def output_predictions(self, y_preds: np.ndarray, csv_out: str) -> None:
        """
        Writes predictions to file.
        Writes predictions made on the dataset to a specified file.

        Parameters
        ----------
        y_preds: np.ndarray
            Predictions to output
        csv_out: str
            Name of file to write predictions to.
        """

        data_ids = self.dataset.ids
        n_tasks = self.dataset.n_tasks
        y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
        assert len(y_preds) == len(data_ids)
        with open(csv_out, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["ID"] + list(self.dataset.label_names))
            for mol_id, y_pred in zip(data_ids, y_preds):
                csvwriter.writerow([mol_id] + list(y_pred))

    def compute_model_performance(self,
                                  metrics: Union[Metric, List[Metric]],
                                  per_task_metrics: bool = False) -> Tuple[Dict, Dict]:
        """
        Computes statistics of model on test data and saves results to csv.

        Parameters
        ----------
        metrics: Union[Metric, List[Metric]]
            The set of metrics provided.
            If a single `Metric` object is provided or a list is provided, it will evaluate `Model` on those metrics.
        per_task_metrics: bool
            If True, return computed metric for each task on multitask dataset.

        Returns
        -------
        multitask_scores: dict
            Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict
            If `per_task_metrics == True`, then returns a second dictionary of scores for each task separately.
        """
        n_tasks = self.dataset.n_tasks
        y = self.dataset.y
        y_pred = self.model.predict(self.dataset)
        if not y.shape == np.array(y_pred).shape:
            y_pred = normalize_labels_shape(y_pred, n_tasks)

        multitask_scores = {}
        all_task_scores = {}

        # Compute multitask metrics
        for metric in metrics:
            results = metric.compute_metric(y, y_pred, per_task_metrics=per_task_metrics, n_tasks=n_tasks)
            if per_task_metrics:
                multitask_scores[metric.name], computed_metrics = results
                all_task_scores[metric.name] = computed_metrics
            else:
                multitask_scores[metric.name], all_task_scores = results

        if not per_task_metrics:
            return multitask_scores, {}
        else:
            return multitask_scores, all_task_scores
