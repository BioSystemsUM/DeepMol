import os
from datetime import datetime
from typing import List, Tuple, Union, Dict

import numpy as np

from deepmol.base import Transformer, Predictor
from deepmol.datasets import Dataset
from deepmol.metrics import Metric


class Pipeline(Transformer):
    """
    Pipeline of transformers and predictors. The last step must be a predictor, all other steps must be
    transformers. It applies a list of transformers in a sequence followed (or not) by a predictor.
    The transformers must implement the fit() and transform() methods, the predictor must implement the
    fit() and predict() methods.
    """

    def __init__(self, steps: List[Tuple[str, Union[Transformer, Predictor]]], path: str = None) -> None:
        """
        Pipeline of transformers and predictors. The last step must be a predictor, all other steps must be
        transformers. It applies a list of transformers in a sequence followed by a predictor.

        Parameters
        ----------
        steps: List[Tuple[str, Union[Transformer, Predictor]]]
            List of (name, transformer/predictor) tuples that are applied sequentially to the data.
        path: str
            Path to directory where pipeline will be stored. If not specified, pipeline will be stored in a temporary
            directory.
        """
        super().__init__()
        self.steps = steps
        self.path = path

    @property
    def is_fitted(self) -> bool:
        """
        Whether the pipeline is fitted.

        Returns
        -------
        is_fitted: bool
            Whether the pipeline is fitted.
        """
        return all([step[1].is_fitted for step in self.steps])

    def is_prediction_pipeline(self) -> bool:
        """
        Whether the pipeline is a prediction pipeline.

        Returns
        -------
        is_prediction_pipeline: bool
            Whether the pipeline is a prediction pipeline.
        """
        return isinstance(self.steps[-1][1], Predictor)

    def _validate_steps(self) -> None:
        """
        Validate the pipeline steps.
        A pipeline must consist of a sequence of transformers followed (or not) by a predictor.

        Raises
        ------
        ValueError
            If the pipeline steps are not valid.
        """
        if len(self.steps) < 1:
            raise ValueError("Pipeline must have at least one step.")
        names, steps = zip(*self.steps)
        if len(set(names)) != len(names):
            raise ValueError("Pipeline steps must have unique names.")
        if len(self.steps) > 1:
            if not all([isinstance(step, Transformer) for step in steps[:-1]]):
                raise ValueError("All steps except the last one must be transformers.")

    def _set_paths(self):
        """
        It sets the paths for the transformers and predictor.
        """
        if self.path is None:
            current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            self.path = os.path.join(os.getcwd(), f'pipeline-{current_time}')

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if self.is_prediction_pipeline():
            self.steps[-1][1].model_dir = self.path

    def fit(self, dataset_train: Dataset) -> 'Pipeline':
        """
        Fit the pipeline to the train data.

        Parameters
        ----------
        dataset_train: Dataset
            Dataset to fit the pipeline to.

        Returns
        -------
        self: Pipeline
            Fitted pipeline.
        """
        self._fit(dataset_train)
        return self

    def _fit(self, dataset: Dataset) -> 'Pipeline':
        """
        Fit the pipeline to the dataset. It also validates if the pipeline steps make sense.

        Parameters
        ----------
        dataset: Dataset
            Dataset to fit the pipeline to.

        Returns
        -------
        self: Pipeline
            Fitted pipeline.
        """
        self._validate_steps()

        if self.is_prediction_pipeline():
            for name, transformer in self.steps[:-1]:
                dataset = transformer.fit_transform(dataset)
            self.steps[-1][1].fit(dataset)
        else:
            for name, transformer in self.steps:
                dataset = transformer.fit_transform(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset using the pipeline transformers.

        Parameters
        ----------
        dataset: Dataset
            Dataset to transform.

        Returns
        -------
        dataset: Dataset
            Transformed dataset.
        """
        if self.is_prediction_pipeline:
            steps = self.steps[:-1]
        else:
            steps = self.steps

        for step in steps:
            dataset = step[1].transform(dataset)
        return dataset

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Make predictions on a dataset using the pipeline predictor.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make predictions on.

        Returns
        -------
        y_pred: np.ndarray
            Predictions.
        """
        if not self.is_prediction_pipeline:
            raise ValueError("Pipeline is not a prediction pipeline.")
        dataset = self.transform(dataset)
        y_pred = self.steps[-1][1].predict(dataset)
        return y_pred

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Make predictions on a dataset using the pipeline predictor.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make predictions on.

        Returns
        -------
        y_pred: np.ndarray
            Predictions.
        """
        if not self.is_prediction_pipeline:
            raise ValueError("Pipeline is not a prediction pipeline.")
        dataset = self.transform(dataset)
        y_pred = self.steps[-1][1].predict_proba(dataset)
        return y_pred

    def evaluate(self,
                 dataset: Dataset,
                 metrics: Union[List[Metric]],
                 per_task_metrics: bool = False) -> Tuple[Dict, Union[None, Dict]]:
        """
        Evaluate the pipeline on a dataset based on the provided metrics.

        Parameters
        ----------
        dataset: Dataset
            Dataset to evaluate on.
        metrics: Union[List[Metric]]
            List of metrics to evaluate on.
        per_task_metrics: bool
            Whether to return per-task metrics.

        Returns
        -------
        multitask_scores: dict
            Dictionary mapping names of metrics to metric scores.
        all_task_scores: dict
            If `per_task_metrics == True` is passed as a keyword argument, then returns a second dictionary of scores
            for each task separately.
        """
        if not self.is_prediction_pipeline:
            raise ValueError("Pipeline is not a prediction pipeline.")
        dataset = self.transform(dataset)
        return self.steps[-1][1].evaluate(dataset, metrics, per_task_metrics)

    def save(self):
        """
        Save the pipeline to disk.
        """
        self._set_paths()
        for name, transformer in self.steps[:-1]:
            transformer.to_pickle(self.path)
        if self.is_prediction_pipeline():
            self.steps[-1][1].save()
        else:
            self.steps[-1][1].to_pickle(self.path)

    @classmethod
    def load(cls, path: str) -> 'Pipeline':
        pass