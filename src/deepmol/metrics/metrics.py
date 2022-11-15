from typing import Tuple, Any, Union, List

import numpy as np

from deepmol.utils.utils import normalize_labels_shape


class Metric(object):
    """Class for computing machine learning metrics.

    Metrics can be imported from scikit-learn or can be user defined functions.
    """

    def __init__(self,
                 metric: callable,
                 task_averager: callable = None,
                 name: str = None,
                 mode: str = None,
                 n_tasks: int = None,
                 classification_handling_mode: str = None,
                 threshold_value: float = None):
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

    def compute_metric(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       n_tasks: int = None,
                       n_classes: int = 2,
                       per_task_metrics: bool = False,
                       **kwargs) -> Tuple[Any, Union[float, List[float]]]:
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
        n_classes: int
            Number of classes in data for classification tasks.
        per_task_metrics: bool
            If true, return computed metric for each task on multitask dataset.
        kwargs: dict
            Will be passed on to self.metric

        Returns
        -------
        Tuple[Any, Union[float, List[float]]]
            Tuple with the task averager computed value and a numpy array containing metric values for each task.
        """

        if n_tasks is None:
            if self.n_tasks is None and isinstance(y_true, np.ndarray):
                if len(y_true.shape) == 1:
                    n_tasks = 1
                elif len(y_true.shape) >= 2:
                    n_tasks = y_true.shape[1]
            else:
                n_tasks = self.n_tasks

        # TODO: Needs to be specified to deal with multitasking
        # y_true = normalize_labels_shape(y_true, mode=self.mode, n_tasks=n_tasks, n_classes=n_classes)
        # y_pred = normalize_prediction_shape(y_pred, mode=self.mode, n_tasks=n_tasks, n_classes=n_classes)
        # if self.mode == "classification":
        #    y_true = handle_classification_mode(y_true, self.classification_handling_mode, self.threshold_value)
        #    y_pred = handle_classification_mode(y_pred, self.classification_handling_mode, self.threshold_value)

        # n_samples = y_true.shape[0]

        computed_metrics = []

        # assuming this is provided with values per column for each task
        # TODO: check this out (needs to be changed to deal with multitasking)
        # print(y_true)
        # print(y_pred)
        for task in range(n_tasks):
            y_task = y_true  # [:, task]
            y_pred_task = y_pred  # [:, task]

            metric_value = self.compute_singletask_metric(y_task,
                                                          y_pred_task,
                                                          **kwargs)
            computed_metrics.append(metric_value)
        print(str(self.metric.__name__) + ': \n', computed_metrics[0])
        if n_tasks == 1:
            computed_metrics = computed_metrics[0]  # type: ignore

        if not per_task_metrics:
            try:
                return self.task_averager(computed_metrics)
            except Exception as e:
                print('WARNING: task averager ', e)
        else:
            return self.task_averager(computed_metrics), computed_metrics

    # TODO: implement to multitask
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

        if self.mode == "regression":
            if len(y_true.shape) != 1 or len(y_pred.shape) != 1 or len(y_true) != len(y_pred):
                raise ValueError("For regression metrics, y_true and y_pred must both be of shape (N,)")

        elif self.mode == "classification":
            pass

        else:
            raise ValueError("Only classification and regression are supported for metrics calculations.")

        try:
            metric_value = self.metric(y_true, y_pred, **kwargs)
        except Exception as e:
            # deal with different shapes of the otput of predict and predict_proba
            if len(y_pred.shape) == 3:  # output of the deepchem MultitaskClassifier model
                y_pred_2 = []
                for p in y_pred:
                    y_pred_2.append(p[0])
                y_pred_mod = normalize_labels_shape(y_pred_2)
                metric_value = self.metric(y_true, y_pred_mod, **kwargs)
            else:
                y_pred_mod = normalize_labels_shape(y_pred)
                metric_value = self.metric(y_true, y_pred_mod, **kwargs)
        return metric_value
