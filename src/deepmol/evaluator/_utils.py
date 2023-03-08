from typing import Union, List

from deepmol.metrics import Metric


def _process_metric_input(metrics: Union[Metric, callable, List[Union[Metric, callable]]]) -> List[Metric]:
    """
    Method which processes metrics correctly.
    Metrics can be `metrics.Metric` or callables or o list of those . Metric functions are functions which
    accept two arguments `y_true, y_pred` both of which must be `np.ndarray` objects and return a float value. This
    functions normalizes these different types of inputs to type `list[metrics.Metric]` object for ease of later
    processing.

    Parameters
    ----------
    metrics: Union[Metric, callable, List[Union[Metric, callable]]]
        The metrics to process.

    Returns
    -------
    final_metrics: list[Metric]
        Converts all input metrics and outputs a list of `Metric` objects.
    """
    # Make sure input is a list
    if not isinstance(metrics, list):
        metrics = [metrics]
    final_metrics = []
    for i, metric in enumerate(metrics):
        if isinstance(metric, Metric):
            final_metrics.append(metric)
        elif callable(metric):
            wrap_metric = Metric(metric, name="metric-%d" % (i + 1))
            final_metrics.append(wrap_metric)
        else:
            raise ValueError("Metrics must be Metric objects or callables.")
    return final_metrics
