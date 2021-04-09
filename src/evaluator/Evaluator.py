import numpy as np
import csv

from typing import Optional, Union, Tuple, Dict, List, Iterable, Any
Score = Dict[str, float]

from Datasets.Datasets import Dataset
from splitters.splitters import RandomSplitter
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
            classification model.
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
                                  metrics: Metric,
                                  per_task_metrics: bool = False,
                                  n_classes: int = 2) -> Union[Score, Tuple[Score, Score]]:
        """
        Computes statistics of model on test data and saves results to csv.

        Parameters
        ----------
        metrics: Metric/list[Metric]
            The set of metrics provided. If a single `Metric`
            object is provided or a list is provided, it will evaluate
            `Model` on these metrics.

        per_task_metrics: bool, optional
            If true, return computed metric for each task on multitask dataset.

        n_classes: int, optional (default None)
            If specified, will use `n_classes` as the number of unique classes
            in the `Dataset`.

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
            return multitask_scores
        else:
            return multitask_scores, all_task_scores


class GeneratorEvaluator(object):
    #TODO: comments
    '''Evaluate models on a stream of data.
    This class is a partner class to `Evaluator`. Instead of operating
    over datasets this class operates over a generator which yields
    batches of data to feed into provided model.
    Examples
    --------
    # import deepchem as dc
    # import numpy as np
    # X = np.random.rand(10, 5)
    # y = np.random.rand(10, 1)
    # dataset = dc.data.NumpyDataset(X, y)
    # model = dc.models.MultitaskRegressor(1, 5)
    # generator = model.default_generator(dataset, pad_batches=False)
    # transformers = []
    # Then you can evaluate this model as follows
    # import sklearn
    # evaluator = GeneratorEvaluator(model, generator, transformers)
    # multitask_scores = evaluator.compute_model_performance(sklearn.metrics.mean_absolute_error)
    Evaluators can also be used with `dc.metrics.Metric` objects as well
    in case you want to customize your metric further. (Note that a given
    generator can only be used once so we have to redefine the generator here.)
    # generator = model.default_generator(dataset, pad_batches=False)
    # evaluator = GeneratorEvaluator(model, generator, transformers)
    # metric = dc.metrics.Metric(dc.metrics.mae_score)
    # multitask_scores = evaluator.compute_model_performance(metric)
    '''

    def __init__(self,
                 model,
                 generator: Iterable[Tuple[Any, Any, Any]],
                 labels: Optional[List] = None):#,
                 #weights: Optional[List] = None):
        """
        Parameters
        ----------
        model: Model
          Model to evaluate.
        generator: generator
          Generator which yields batches to feed into the model. For a
          KerasModel, it should be a tuple of the form (inputs, labels,
          weights). The "correct" way to create this generator is to use
          `model.default_generator` as shown in the example above.
        labels: list of Layer
          layers which are keys in the generator to compare to outputs
        weights: list of Layer
          layers which are keys in the generator for weight matrices
        """

        self.model = model
        self.generator = generator
        #self.output_transformers = [
        #    transformer for transformer in transformers if transformer.transform_y
        #]
        self.label_keys = labels
        #self.weights = weights
        if labels is not None and len(labels) != 1:
            raise ValueError("GeneratorEvaluator currently only supports one label")

    def compute_model_performance(self,
                                  metrics: Metric,
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
          `np.ndarray` objects and return a floating point score.
        per_task_metrics: bool, optional
          If true, return computed metric for each task on multitask
          dataset.
        use_sample_weights: bool, optional (default False)
          If set, use per-sample weights `w`.
        n_classes: int, optional (default None)
          If specified, will assume that all `metrics` are classification
          metrics and will use `n_classes` as the number of unique classes
          in `self.dataset`.
        Returns
        -------
        multitask_scores: dict
          Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict, optional
          If `per_task_metrics == True`, then returns a second dictionary
          of scores for each task separately.
        """
        metrics = _process_metric_input(metrics)

        # We use y/w to aggregate labels/weights across generator.
        y = []
        #w = []

        # GENERATOR CLOSURE
        def generator_closure():
            """This function is used to pull true labels/weights out as we iterate over the generator."""
            if self.label_keys is None:
                #weights = None
                # This is a KerasModel.
                for batch in self.generator:
                    # Some datasets have weights
                    try:
                        #inputs, labels, weights = batch
                        inputs, labels = batch
                    except ValueError:
                        try:
                            #inputs, labels, weights, ids = batch
                            inputs, labels, ids = batch
                        except ValueError:
                            raise ValueError(
                                "Generator must yield values of form (input, labels) or (input, labels, ids)"
                            )
                    y.append(labels[0])
                    #if len(weights) > 0:
                    #    w.append(weights[0])
                    #yield (inputs, labels, weights)
                    yield (inputs, labels)

        # Process predictions and populate y/w lists
        y_pred = self.model.predict_on_generator(generator_closure())

        # Combine labels/weights
        y = np.concatenate(y, axis=0)
        #w = np.concatenate(w, axis=0)

        multitask_scores = {}
        all_task_scores = {}

        # Undo data transformations.
        #y = dc.trans.undo_transforms(y, self.output_transformers)
        #y_pred = dc.trans.undo_transforms(y_pred, self.output_transformers)

        # Compute multitask metrics
        for metric in metrics:
            results = metric.compute_metric(y,
                                            y_pred,
                                            #w,
                                            per_task_metrics=per_task_metrics,
                                            n_classes=n_classes,
                                            use_sample_weights=use_sample_weights)
            if per_task_metrics:
                multitask_scores[metric.name], computed_metrics = results
                all_task_scores[metric.name] = computed_metrics
            else:
                multitask_scores[metric.name] = results

        if not per_task_metrics:
            return multitask_scores
        else:
            return multitask_scores, all_task_scores
