import numpy as np
import csv

from typing import Optional, Union, Tuple, Dict, List
Score = Dict[str, float]

from Dataset.Dataset import Dataset
from splitters.splitters import  RandomSplitter
from metrics.Metrics import Metric


def _process_metric_input(metrics: Metric) -> List[Metric]:
    """Method which processes metrics correctly.
    Metrics can be input as `metrics.Metric` objects, lists of
    `metrics.Metric`. Metric functions are functions which accept
    two arguments `y_true, y_pred` both of which must be `np.ndarray`
    objects and return a float value. This functions normalizes these
    different types of inputs to type `list[metrics.Metric]` object
    for ease of later processing.

    Parameters
    ----------
    metrics: metrics.Metric/list[dc.metrics.Metric]
        Input metrics to process.
    Returns
    -------
    final_metrics: list[metrics.Metric]
        Converts all input metrics and outputs a list of
        `metrics.Metric` objects.
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
            raise ValueError("Metrics must be metrics.Metric objects.")
    return final_metrics

class Evaluator(object):
    """Class that evaluates a model on a given dataset.
    The evaluator class is used to evaluate a `Model` class on
    a given `Dataset` object.
    """

    def __init__(self, model, dataset: Dataset):#, metric: Metric):

        """Initialize this evaluator
        Parameters
        ----------
        model: Model
        Model to evaluate. Note that this must be a regression or
        classification model and not a generative model.
        dataset: Dataset
        Dataset object to evaluate `model` on.
        """

        self.model = model
        self.dataset = dataset

    def output_statistics(self, scores: Score, stats_out: str):
        """ Write computed stats to file.
        Parameters
        ----------
        scores: dict
        Dictionary mapping names of metrics to scores.
        stats_out: str
        Name of file to write scores to.
        """
        with open(stats_out, "w") as statsfile:
            statsfile.write(str(scores) + "\n")

    def output_predictions(self, y_preds: np.ndarray, csv_out: str):
        """Writes predictions to file.
        Writes predictions made on `self.dataset` to a specified file on
        disk. `self.dataset.ids` are used to format predictions.
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

    def compute_model_performance(
        self,
        metrics: Metric,
        csv_out: Optional[str] = None,
        stats_out: Optional[str] = None,
        per_task_metrics: bool = False,
        use_sample_weights: bool = False,
        n_classes: int = 2) -> Union[Score, Tuple[Score, Score]]:
        """
        Computes statistics of model on test data and saves results to csv.
        Parameters
        ----------
        metrics: dc.metrics.Metric/list[dc.metrics.Metric]/function
        The set of metrics provided. This class attempts to do some
        intelligent handling of input. If a single `dc.metrics.Metric`
        object is provided or a list is provided, it will evaluate
        `self.model` on these metrics. If a function is provided, it is
        assumed to be a metric function that this method will attempt to
        wrap in a `dc.metrics.Metric` object. A metric function must
        accept two arguments, `y_true, y_pred` both of which are
        `np.ndarray` objects and return a floating point score. The
        metric function may also accept a keyword argument
        `sample_weight` to account for per-sample weights.
        csv_out: str, optional (DEPRECATED)
        Filename to write CSV of model predictions.
        stats_out: str, optional (DEPRECATED)
        Filename to write computed statistics.
        per_task_metrics: bool, optional
        If true, return computed metric for each task on multitask dataset.
        use_sample_weights: bool, optional (default False)
        If set, use per-sample weights `w`.
        n_classes: int, optional (default None)
        If specified, will use `n_classes` as the number of unique classes
        in `self.dataset`. Note that this argument will be ignored for
        regression metrics.
        Returns
        -------
        multitask_scores: dict
        Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict, optional
        If `per_task_metrics == True`, then returns a second dictionary
        of scores for each task separately.
        """
        
         # Process input metrics
        metrics = _process_metric_input(metrics)

        y = self.dataset.y

        y_pred = self.model.predict(self.dataset)
        n_tasks = len(self.dataset.get_task_names())

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
            return multitask_scores
        else:
            return multitask_scores, all_task_scores


