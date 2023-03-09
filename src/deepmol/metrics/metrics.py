from typing import Tuple, Any, Union, List

import numpy as np

from deepmol.loggers import Logger
from deepmol.utils.utils import normalize_labels_shape


class Metric(object):
    """
    Class for computing machine learning metrics.

    Metrics can be imported from scikit-learn or can be user defined functions.
    """

    def __init__(self,
                 metric: callable,
                 task_averager: callable = None,
                 name: str = None,
                 mode: str = None,
                 n_tasks: int = None,
                 classification_handling_mode: str = None,
                 threshold_value: float = None,
                 **kwargs) -> None:
        # TODO: review threshold values and change to deal with a variable threshold (i.e. different from 0.5)
        """
        Parameters
        ----------
        metric: callable
            Callable that takes args y_true, y_pred (in that order) and computes desired score.
        task_averager: callable
            If not None, should be a function that averages metrics across tasks.
        name: str
            Name of this metric
        mode: str
            Should be "classification" or "regression."
        n_tasks: int
            The number of tasks this class is expected to handle.
        classification_handling_mode: str
            Models by default predict class probabilities for classification problems. This means that for a given
            singletask prediction, after shape normalization, the prediction will be a numpy array of shape
            `(N, n_classes)` with class probabilities.
            It can take on the following values:
            - None: default value. Pass in `y_pred` directy into the metric.
            - "threshold": Use `threshold_predictions` to threshold `y_pred`. Use `threshold_value` as the desired
            threshold.
            - "threshold-one-hot": Use `threshold_predictions` to threshold `y_pred` using `threshold_values`, then
            apply `to_one_hot` to output.
        threshold_value: float
            If set, and `classification_handling_mode` is "threshold" or "threshold-one-hot" apply a thresholding
            operation to values with this threshold.
        kwargs:
            Additional arguments to pass to metric.
        """
        self.metric = metric
        if task_averager is None:
            self.task_averager = np.mean
        else:
            self.task_averager = task_averager

        if name is None:
            if task_averager is None:
                if hasattr(self.metric, '__name__'):
                    self.name = self.metric.__name__
                else:
                    self.name = "unknown metric"
            else:
                if hasattr(self.metric, '__name__'):
                    self.name = task_averager.__name__ + "-" + self.metric.__name__
                else:
                    self.name = "unknown metric"
        else:
            self.name = name
        if mode is None:
            # Some default metrics
            if self.metric.__name__ in ["roc_auc_score",
                                        "matthews_corrcoef",
                                        "recall_score",
                                        "accuracy_score",
                                        "kappa_score",
                                        "cohen_kappa_score",
                                        "precision_score",
                                        "balanced_accuracy_score",
                                        "prc_auc_score",
                                        "f1_score",
                                        "bedroc_score",
                                        "jaccard_score",
                                        "jaccard_index",
                                        "pixel_error",
                                        "confusion_matrix",
                                        "classification_report"]:
                mode = "classification"
                # Defaults sklearn's metrics with required behavior
                if classification_handling_mode is None:
                    if self.metric.__name__ in ["matthews_corrcoef", "cohen_kappa_score", "kappa_score",
                                                "balanced_accuracy_score", "recall_score", "jaccard_score",
                                                "jaccard_index", "pixel_error", "f1_score"]:

                        classification_handling_mode = "threshold"

                    elif self.metric.__name__ in ["accuracy_score", "precision_score", "bedroc_score"]:
                        classification_handling_mode = "threshold-one-hot"

                    elif self.metric.__name__ in ["roc_auc_score", "prc_auc_score"]:
                        classification_handling_mode = None

            elif self.metric.__name__ in ["pearson_r2_score", "r2_score", "mean_squared_error",
                                          "mean_absolute_error", "rms_score", "mae_score", "pearsonr",
                                          "median_absolute_error"]:
                mode = "regression"
            else:
                raise ValueError(
                    "Please specify the mode of this metric. mode must be 'regression' or 'classification'")

        self.mode = mode
        self.n_tasks = n_tasks

        if classification_handling_mode not in [None, "threshold", "threshold-one-hot"]:
            raise ValueError("classification_handling_mode must be one of None, 'threshold', 'threshold_one_hot'")

        self.classification_handling_mode = classification_handling_mode
        self.threshold_value = threshold_value
        self.kwargs = kwargs
        self.logger = Logger()

    def compute_metric(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       n_tasks: int,
                       per_task_metrics: bool = False) -> Tuple[Any, List[Any]]:
        """
        Compute a performance metric for each task.

        Parameters
        ----------
        y_true: np.ndarray
            An np.ndarray containing true values for each task.
        y_pred: np.ndarray
            An np.ndarray containing predicted values for each task.
        n_tasks: int
            The number of tasks this class is expected to handle.
        per_task_metrics: bool
            If true, return computed metric for each task on multitask dataset.
        kwargs: dict
            Will be passed on to self.metric

        Returns
        -------
        Tuple[Any, List[Any]]
            Tuple with the task averager computed value and a list containing metric values for each task.
        """
        try:
            y_task = y_true
            y_pred_task = y_pred

            metric_value = self.compute_singletask_metric(y_task,
                                                          y_pred_task,
                                                          **self.kwargs)
            cm = metric_value
        except ValueError as e:
            cm = None
        if n_tasks > 1 and per_task_metrics:
            computed_metrics = []
            for task in range(n_tasks):
                y_task = y_true[:, task]
                y_pred_task = y_pred[:, task]

                metric_value = self.compute_singletask_metric(y_task,
                                                              y_pred_task)
                computed_metrics.append(metric_value)
            if cm is None:
                cm = self.task_averager(computed_metrics)
            return cm, computed_metrics
        return cm, []

    def compute_singletask_metric(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  **kwargs) -> float:
        """
        Compute a metric value for a singletask problem.

        Parameters
        ----------
        y_true: `np.ndarray`
            True values array.
        y_pred: `np.ndarray`
            Predictions array.
        kwargs: dict
            Will be passed on to self.metric

        Returns
        -------
        metric_value: float
            The computed value of the metric.
        """
        # deal with metrics that require 0 or 1 values not probabilities
        try:
            metric_value = self.metric(y_true, y_pred, **kwargs)
        except ValueError as e:
            # round values in y_ped to 0 or 1
            y_pred_rounded = np.round(y_pred)
            metric_value = self.metric(y_true, y_pred_rounded, **kwargs)
        return metric_value
