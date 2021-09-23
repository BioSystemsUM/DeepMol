from typing import Optional, List

import numpy as np

from src.Datasets.Datasets import Dataset
from src.evaluator.Evaluator import Evaluator
from src.metrics.Metrics import Metric
from src.utils.utils import normalize_labels_shape


class VotingEnsemble(object):
    """
    An ensemble meta-estimator that combines the predictions of
    various models ('KerasModel' and/or 'DeepChemModel' objects).
    For classification tasks, this is a majority voting ensemble.
    For regression tasks, it averages the individual predictions
    to obtain the final prediction.
    """

    # it could inherit from Model, but that might be confusing as self.model would be a list of Model objects instead of a single model

    def __init__(self, base_models: List,
                 featurizers: List,
                 mode: Optional[str] = 'classification',
                 model_dir: Optional[str] = None) -> None:
        self.base_models = base_models
        self.featurizers = featurizers
        self.mode = mode
        self.model_dir = model_dir

    def fit(self, dataset: Dataset):
        """Fits the models on data.

        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        """
        fit_models = []
        for model, featurizer in zip(self.base_models, self.featurizers):
            featurized_dataset = featurizer.featurize(dataset)
            model.fit(featurized_dataset)
            fit_models.append(model)
            del featurized_dataset
        self.base_models = fit_models

    def predict(self, dataset: Dataset):
        """Makes predictions for the provided dataset.
        Parameters
        ----------
        dataset: Dataset
          Dataset to make prediction on.
        Returns
        -------
        np.ndarray
          Values predicted by the ensemble.
        """
        predictions = []
        for model, featurizer in zip(self.base_models, self.featurizers):
            featurized_dataset = featurizer.featurize(dataset)
            y_pred = model.predict(featurized_dataset)
            if self.mode == 'classification':
                y_pred = normalize_labels_shape(y_pred)
            predictions.append(y_pred)
            del featurized_dataset

        if self.mode == 'classification':  # classification - majority class
            ensemble_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                                axis=0, arr=np.array(predictions))
        else:  # regression - ensemble prediction is the average value of predictions
            ensemble_pred = np.mean(np.array(predictions), axis=0)
        return ensemble_pred

    def evaluate(self,
                 dataset: Dataset,
                 metrics: List[Metric],
                 per_task_metrics: bool = False,
                 n_classes: int = 2):
        """
        Evaluates the performance of this ensemble on the specified dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        metrics: Metric / List[Metric]
            The set of metrics provided.
        per_task_metrics: bool, optional (default False)
            If true, return computed metric for each task on multitask dataset.
        n_classes: int, optional (default None)
            If specified, will use `n_classes` as the number of unique classes.

        Returns
        -------
        multitask_scores: dict
            Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict, optional
            If `per_task_metrics == True` is passed as a keyword argument,
            then returns a second dictionary of scores for each task
            separately.
        """

        evaluator = Evaluator(self, dataset) # passing this class (self) as the model. It will call the VotingEnsemble predict method

        return evaluator.compute_model_performance(metrics,
                                                   per_task_metrics=per_task_metrics,
                                                   n_classes=n_classes)

    def save(self):
        """Saves all of the base models in the same directory (model_dir)"""
        # KerasModel currently doesn't have the save (or reload) method implemented
        for model in self.base_models:
            model.model_dir = self.model_dir # to guarantee that all models are saved in the same model_dir
            model.save()

    def reload(self):
        pass
