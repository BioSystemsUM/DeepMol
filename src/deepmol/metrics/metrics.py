from typing import Tuple, Any, List

import numpy as np

from deepmol.loggers import Logger


class Metric(object):
    """
    Class for computing machine learning metrics.

    Metrics can be imported from scikit-learn or can be user defined functions.
    """

    def __init__(self,
                 metric: callable,
                 task_averager: callable = None,
                 name: str = None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        metric: callable
            Callable that takes args y_true, y_pred (in that order) and computes desired score.
        task_averager: callable
            If not None, should be a function that averages metrics across tasks.
        name: str
            Name of this metric
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
                self.name = self.metric.__name__
            else:
                self.name = self.task_averager.__name__ + "-" + self.metric.__name__
        else:
            self.name = name
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
            try:
                # round values in y_ped to 0 or 1
                y_pred_rounded = np.round(y_pred)
                metric_value = self.metric(y_true, y_pred_rounded, **kwargs)
            except ValueError as e:
                # transform y_pred [[0, 1], [1, 0]] to [1, 0]
                y_pred_single = np.argmax(y_pred, axis=1)
                metric_value = self.metric(y_true, y_pred_single, **kwargs)
        return metric_value
