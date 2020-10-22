class Metric(object):
  """Wrapper class for computing user-defined metrics.
  The `Metric` class provides a wrapper for standardizing the API
  around different classes of metrics that may be useful for DeepChem
  models. The implementation provides a few non-standard conveniences
  such as built-in support for multitask and multiclass metrics.
  There are a variety of different metrics this class aims to support.
  Metrics for classification and regression that assume that values to
  compare are scalars are supported.
  At present, this class doesn't support metric computation on models
  which don't present scalar outputs. For example, if you have a
  generative model which predicts images or molecules, you will need
  to write a custom evaluation and metric setup.
  """

  def __init__(self,
               metric: Callable[..., float],
               task_averager: Optional[Callable[..., Any]] = None,
               name: Optional[str] = None,
               threshold: Optional[float] = None,
               mode: Optional[str] = None,
               n_tasks: Optional[int] = None,
               classification_handling_mode: Optional[str] = None,
               threshold_value: Optional[float] = None,
               compute_energy_metric: Optional[bool] = None):
    """
    Parameters
    ----------
    metric: function
      Function that takes args y_true, y_pred (in that order) and
      computes desired score. If sample weights are to be considered,
      `metric` may take in an additional keyword argument
      `sample_weight`.
    task_averager: function, default None
      If not None, should be a function that averages metrics across
      tasks.
    name: str, default None
      Name of this metric
    threshold: float, default None (DEPRECATED)
      Used for binary metrics and is the threshold for the positive
      class.
    mode: str, default None
      Should usually be "classification" or "regression."
    n_tasks: int, default None
      The number of tasks this class is expected to handle.
    classification_handling_mode: str, default None
      DeepChem models by default predict class probabilities for
      classification problems. This means that for a given singletask
      prediction, after shape normalization, the DeepChem prediction will be a
      numpy array of shape `(N, n_classes)` with class probabilities.
      `classification_handling_mode` is a string that instructs this method
      how to handle transforming these probabilities. It can take on the
      following values:
      - None: default value. Pass in `y_pred` directy into `self.metric`.
      - "threshold": Use `threshold_predictions` to threshold `y_pred`. Use
        `threshold_value` as the desired threshold.
      - "threshold-one-hot": Use `threshold_predictions` to threshold `y_pred`
        using `threshold_values`, then apply `to_one_hot` to output.
    threshold_value: float, default None
      If set, and `classification_handling_mode` is "threshold" or
      "threshold-one-hot" apply a thresholding operation to values with this
      threshold. This option is only sensible on binary classification tasks.
      If float, this will be applied as a binary classification value.
    compute_energy_metric: bool, default None (DEPRECATED)
      Deprecated metric. Will be removed in a future version of
      DeepChem. Do not use.
    """
    #if threshold is not None:
    #  logger.warn(
    #      "threshold is deprecated and will be removed in a future version of DeepChem."
    #      "Set threshold in compute_metric instead.")

    if compute_energy_metric is not None:
      self.compute_energy_metric = compute_energy_metric
    #  logger.warn(
    #      "compute_energy_metric is deprecated and will be removed in a future version of DeepChem."
    #  )
    else:
      self.compute_energy_metric = False

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
      # These are some smart defaults
      if self.metric.__name__ in [
          "roc_auc_score",
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
      ]:
        mode = "classification"
        # These are some smart defaults corresponding to sklearn's required
        # behavior
        if classification_handling_mode is None:
          if self.metric.__name__ in [
              "matthews_corrcoef", "cohen_kappa_score", "kappa_score",
              "balanced_accuracy_score", "recall_score", "jaccard_score",
              "jaccard_index", "pixel_error", "f1_score"
          ]:
            classification_handling_mode = "threshold"
          elif self.metric.__name__ in [
              "accuracy_score", "precision_score", "bedroc_score"
          ]:
            classification_handling_mode = "threshold-one-hot"
          elif self.metric.__name__ in ["roc_auc_score", "prc_auc_score"]:
            classification_handling_mode = None
      elif self.metric.__name__ in [
          "pearson_r2_score", "r2_score", "mean_squared_error",
          "mean_absolute_error", "rms_score", "mae_score", "pearsonr"
      ]:
        mode = "regression"
      else:
        raise ValueError(
            "Please specify the mode of this metric. mode must be 'regression' or 'classification'"
        )

    self.mode = mode
    self.n_tasks = n_tasks
    if classification_handling_mode not in [
        None, "threshold", "threshold-one-hot"
    ]:
      raise ValueError(
          "classification_handling_mode must be one of None, 'threshold', 'threshold_one_hot'"
      )
    self.classification_handling_mode = classification_handling_mode
    self.threshold_value = threshold_value

  def compute_metric(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     w: Optional[np.ndarray] = None,
                     n_tasks: Optional[int] = None,
                     n_classes: int = 2,
                     filter_nans: bool = False,
                     per_task_metrics: bool = False,
                     use_sample_weights: bool = False,
                     **kwargs) -> np.ndarray:
    """Compute a performance metric for each task.
    Parameters
    ----------
    y_true: np.ndarray
      An np.ndarray containing true values for each task. Must be of shape
      `(N,)` or `(N, n_tasks)` or `(N, n_tasks, n_classes)` if a
      classification metric. If of shape `(N, n_tasks)` values can either be
      class-labels or probabilities of the positive class for binary
      classification problems. If a regression problem, must be of shape
      `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` if a regression metric.
    y_pred: np.ndarray
      An np.ndarray containing predicted values for each task. Must be
      of shape `(N, n_tasks, n_classes)` if a classification metric,
      else must be of shape `(N, n_tasks)` if a regression metric.
    w: np.ndarray, default None
      An np.ndarray containing weights for each datapoint. If
      specified,  must be of shape `(N, n_tasks)`.
    n_tasks: int, default None
      The number of tasks this class is expected to handle.
    n_classes: int, default 2
      Number of classes in data for classification tasks.
    filter_nans: bool, default False (DEPRECATED)
      Remove NaN values in computed metrics
    per_task_metrics: bool, default False
      If true, return computed metric for each task on multitask dataset.
    use_sample_weights: bool, default False
      If set, use per-sample weights `w`.
    kwargs: dict
      Will be passed on to self.metric
    Returns
    -------
    np.ndarray
      A numpy array containing metric values for each task.
    """
    # Attempt some limited shape imputation to find n_tasks
    if n_tasks is None:
      if self.n_tasks is None and isinstance(y_true, np.ndarray):
        if len(y_true.shape) == 1:
          n_tasks = 1
        elif len(y_true.shape) >= 2:
          n_tasks = y_true.shape[1]
      else:
        n_tasks = self.n_tasks
    # check whether n_tasks is int or not
    # This is because `normalize_weight_shape` require int value.
    assert isinstance(n_tasks, int)

    y_true = normalize_labels_shape(
        y_true, mode=self.mode, n_tasks=n_tasks, n_classes=n_classes)
    y_pred = normalize_prediction_shape(
        y_pred, mode=self.mode, n_tasks=n_tasks, n_classes=n_classes)
    if self.mode == "classification":
      y_true = handle_classification_mode(
          y_true, self.classification_handling_mode, self.threshold_value)
      y_pred = handle_classification_mode(
          y_pred, self.classification_handling_mode, self.threshold_value)
    n_samples = y_true.shape[0]
    w = normalize_weight_shape(w, n_samples, n_tasks)
    computed_metrics = []
    for task in range(n_tasks):
      y_task = y_true[:, task]
      y_pred_task = y_pred[:, task]
      w_task = w[:, task]

      metric_value = self.compute_singletask_metric(
          y_task,
          y_pred_task,
          w_task,
          use_sample_weights=use_sample_weights,
          **kwargs)
      computed_metrics.append(metric_value)
    logger.info("computed_metrics: %s" % str(computed_metrics))
    if n_tasks == 1:
      # FIXME: Incompatible types in assignment
      computed_metrics = computed_metrics[0]  # type: ignore

    # DEPRECATED. WILL BE REMOVED IN NEXT DEEPCHEM VERSION
    if filter_nans:
      computed_metrics = np.array(computed_metrics)
      computed_metrics = computed_metrics[~np.isnan(computed_metrics)]
    # DEPRECATED. WILL BE REMOVED IN NEXT DEEPCHEM VERSION
    if self.compute_energy_metric:
      force_error = self.task_averager(computed_metrics[1:]) * 4961.47596096
      logger.info("Force error (metric: np.mean(%s)): %f kJ/mol/A" %
                  (self.name, force_error))
      return computed_metrics[0]
    elif not per_task_metrics:
      return self.task_averager(computed_metrics)
    else:
      return self.task_averager(computed_metrics), computed_metrics

  def compute_singletask_metric(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                w: Optional[np.ndarray] = None,
                                n_samples: Optional[int] = None,
                                use_sample_weights: bool = False,
                                **kwargs) -> float:
    """Compute a metric value.
    Parameters
    ----------
    y_true: `np.ndarray`
      True values array. This array must be of shape `(N,
      n_classes)` if classification and `(N,)` if regression.
    y_pred: `np.ndarray`
      Predictions array. This array must be of shape `(N, n_classes)`
      if classification and `(N,)` if regression.
    w: `np.ndarray`, default None
      Sample weight array. This array must be of shape `(N,)`
    n_samples: int, default None (DEPRECATED)
      The number of samples in the dataset. This is `N`. This argument is
      ignored.
    use_sample_weights: bool, default False
      If set, use per-sample weights `w`.
    kwargs: dict
      Will be passed on to self.metric
    Returns
    -------
    metric_value: float
      The computed value of the metric.
    """
    if n_samples is not None:
      logger.warning("n_samples is a deprecated argument which is ignored.")
    # Attempt to convert both into the same type
    if self.mode == "regression":
      if len(y_true.shape) != 1 or len(
          y_pred.shape) != 1 or len(y_true) != len(y_pred):
        raise ValueError(
            "For regression metrics, y_true and y_pred must both be of shape (N,)"
        )
    elif self.mode == "classification":
      pass
      # if len(y_true.shape) != 2 or len(y_pred.shape) != 2 or y_true.shape != y_pred.shape:
      # raise ValueError("For classification metrics, y_true and y_pred must both be of shape (N, n_classes)")
    else:
      raise ValueError(
          "Only classification and regression are supported for metrics calculations."
      )
    if use_sample_weights:
      metric_value = self.metric(y_true, y_pred, sample_weight=w, **kwargs)
    else:
      metric_value = self.metric(y_true, y_pred, **kwargs)
    return metric_value