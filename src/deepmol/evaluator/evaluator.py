import csv
from typing import Dict, Union, List, Tuple

import numpy as np

from deepmol.datasets import Dataset
from deepmol.metrics import Metric

Score = Dict[str, float]


def _process_metric_input(metrics: Union[Metric, List[Metric]]) -> List[Metric]:
    """
    Method which processes metrics correctly.
    Metrics can be input as `metrics.Metric` objects, lists of `metrics.Metric`. Metric functions are functions which
    accept two arguments `y_true, y_pred` both of which must be `np.ndarray` objects and return a float value. This
    functions normalizes these different types of inputs to type `list[metrics.Metric]` object for ease of later
    processing.

    Parameters
    ----------
    metrics: Union[metrics.Metric, List[metrics.Metric]]
        The metrics to process.

    Returns
    -------
    final_metrics: list[Metric]
        Converts all input metrics and outputs a list of `Metric` objects.
    """
    # Make sure input is a list
    if not isinstance(metrics, list):
        metrics = [metrics]  # type: ignore

    final_metrics = []
    for i, metric in enumerate(metrics):

        if isinstance(metric, Metric):
            final_metrics.append(metric)
        elif callable(metric):
            wrap_metric = Metric(metric, name="metric-%d" % (i + 1))
            final_metrics.append(wrap_metric)
        else:
            raise ValueError("Metrics must be Metric objects.")
    return final_metrics


class Evaluator(object):
    """
    Class that evaluates a model on a given dataset.
    The evaluator class is used to evaluate a `Model` class on a given `Dataset` object.
    """

    def __init__(self, model, dataset: Dataset):

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
    def output_statistics(scores: Score, stats_out: str):
        """
        Write computed stats to file.

        Parameters
        ----------
        scores: Score
            Dictionary mapping names of metrics to scores.
        stats_out: str
            Name of file to write scores to.
        """
        with open(stats_out, "w") as statsfile:
            statsfile.write(str(scores) + "\n")

    def output_predictions(self, y_preds: np.ndarray, csv_out: str):
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
        n_tasks = len(self.dataset.get_task_names())
        y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
        assert len(y_preds) == len(data_ids)
        with open(csv_out, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["ID"] + self.dataset.get_task_names())
            for mol_id, y_pred in zip(data_ids, y_preds):
                csvwriter.writerow([mol_id] + list(y_pred))

    # TODO: Works with singletask, check for multitask
    def compute_model_performance(self,
                                  metrics: Union[Metric, List[Metric]],
                                  per_task_metrics: bool = False,
                                  n_classes: int = 2) -> Tuple[Dict, Union[None, Dict]]:
        """
        Computes statistics of model on test data and saves results to csv.

        Parameters
        ----------
        metrics: Union[metrics.Metric, List[metrics.Metric]]
            The set of metrics provided.
            If a single `Metric` object is provided or a list is provided, it will evaluate `Model` on those metrics.
        per_task_metrics: bool
            If True, return computed metric for each task on multitask dataset.
        n_classes: int
            If specified, will use `n_classes` as the number of unique classes in the `Dataset`.

        Returns
        -------
        multitask_scores: dict
            Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict
            If `per_task_metrics == True`, then returns a second dictionary of scores for each task separately.
        """
        # Process input metrics
        metrics = _process_metric_input(metrics)

        y = self.dataset.y

        y_pred = self.model.predict(self.dataset)
        n_tasks = self.dataset.n_tasks

        multitask_scores = {}
        all_task_scores = {}

        # Compute multitask metrics
        for metric in metrics:
            results = metric.compute_metric(y,
                                            y_pred,
                                            per_task_metrics=per_task_metrics,
                                            n_tasks=n_tasks,
                                            n_classes=n_classes)
            if per_task_metrics:
                multitask_scores[metric.name], computed_metrics = results
                all_task_scores[metric.name] = computed_metrics
            else:
                multitask_scores[metric.name] = results

        if not per_task_metrics:
            return multitask_scores, None
        else:
            return multitask_scores, all_task_scores
