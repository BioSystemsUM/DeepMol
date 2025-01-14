import json
import os
import time
from datetime import datetime
from typing import List, Tuple, Union, Dict

import numpy as np

from deepmol.base import Transformer, Predictor
from deepmol.datasets import Dataset
from deepmol.metrics import Metric
from deepmol.parameter_optimization.hyperparameter_optimization import HyperparameterOptimizer
from deepmol.pipeline._utils import _get_predictor_instance


class Pipeline(Transformer):
    """
    Pipeline of transformers and predictors. The last step must be a predictor, all other steps must be
    transformers. It applies a list of transformers in a sequence followed (or not) by a predictor.
    The transformers must implement the fit() and transform() methods, the predictor must implement the
    fit() and predict() methods.
    """

    def __init__(self, steps: List[Tuple[str, Union[Transformer, Predictor]]], path: str = None,
                 hpo: HyperparameterOptimizer = None) -> None:
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
        hpo: HyperparameterOptimizer
            Hyperparameter optimizer to use for hyperparameter optimization. If not specified, no hyperparameter
            optimization will be performed.
        """
        super().__init__()
        self.steps = steps
        self.path = path
        self.hpo = hpo

    def is_fitted(self) -> bool:
        """
        Whether the pipeline is fitted.

        Returns
        -------
        is_fitted: bool
            Whether the pipeline is fitted.
        """
        return all([step[1].is_fitted() for step in self.steps])

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

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset = None) -> 'Pipeline':
        """
        Fit the pipeline to the train data.

        Parameters
        ----------
        train_dataset: Dataset
            Dataset to fit the pipeline to.
        validation_dataset: Dataset
            Dataset to validate the pipeline on if hpo is not None.

        Returns
        -------
        self: Pipeline
            Fitted pipeline.
        """
        self._fit(train_dataset, validation_dataset)
        return self

    def _fit(self, train_dataset: Dataset, validation_dataset: Dataset = None) -> 'Pipeline':
        """
        Fit the pipeline to the dataset. It also validates if the pipeline steps make sense.

        Parameters
        ----------
        train_dataset: Dataset
            Dataset to fit the pipeline to.
        validation_dataset: Dataset
            Dataset to validate the pipeline on if hpo is not None.

        Returns
        -------
        self: Pipeline
            Fitted pipeline.
        """
        self._validate_steps()

        if self.hpo is None:
            if self.is_prediction_pipeline():
                for name, transformer in self.steps[:-1]:
                    train_dataset = transformer.fit_transform(train_dataset)
                    if validation_dataset is not None:
                        validation_dataset = transformer.transform(validation_dataset)
                
                if validation_dataset is not None and self.steps[-1][1].__class__.__name__ == 'KerasModel':
                    self.steps[-1][1].fit(train_dataset, validation_data = validation_dataset)
                else:
                    self.steps[-1][1].fit(train_dataset)
            else:
                for name, transformer in self.steps:
                    train_dataset = transformer.fit_transform(train_dataset)
        else:
            if self.is_prediction_pipeline():
                for name, transformer in self.steps[:-1]:
                    train_dataset = transformer.fit_transform(train_dataset)
                    if validation_dataset is not None:
                        validation_dataset = transformer.transform(validation_dataset)
            else:
                for name, transformer in self.steps:
                    train_dataset = transformer.fit_transform(train_dataset)
                    if validation_dataset is not None:
                        validation_dataset = transformer.transform(validation_dataset)

            best_model, self.hpo_best_hyperparams_, self.hpo_all_results_ = self.hpo.fit(train_dataset,
                                                                                         validation_dataset)
            if self.is_prediction_pipeline():
                self.steps[-1] = ("best_model", best_model)
            else:
                self.steps.append(("best_model", best_model))

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
        if self.is_prediction_pipeline():
            steps = self.steps[:-1]
        else:
            steps = self.steps

        for step in steps:
            dataset = step[1].transform(dataset)
        return dataset

    def predict(self, dataset: Dataset, return_invalid: bool = False) -> np.ndarray:
        """
        Make predictions on a dataset using the pipeline predictor.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make predictions on.

        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        y_pred: np.ndarray
            Predictions.
        """
        if not self.is_prediction_pipeline:
            raise ValueError("Pipeline is not a prediction pipeline.")

        dataset = self.transform(dataset)
        y_pred = self.steps[-1][1].predict(dataset, return_invalid=return_invalid)

        return y_pred

    def predict_proba(self, dataset: Dataset, return_invalid: bool = False) -> np.ndarray:
        """
        Make predictions on a dataset using the pipeline predictor.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make predictions on.
        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        y_pred: np.ndarray
            Predictions.
        """
        if not self.is_prediction_pipeline:
            raise ValueError("Pipeline is not a prediction pipeline.")
        dataset = self.transform(dataset)
        y_pred = self.steps[-1][1].predict_proba(dataset, return_invalid=return_invalid)
        return y_pred

    def evaluate(self,
                 dataset: Dataset,
                 metrics: List[Metric],
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
        Save the pipeline to disk (transformers and predictor).
        The sequence of transformers is saved in a config file. The transformers and predictor are saved in separate
        files. Transformers are saved as pickle files, while the predictor is saved using its own save method.
        """
        self._set_paths()
        steps_to_save = {}
        for i, (name, transformer) in enumerate(self.steps[:-1]):
            transformer_path = os.path.join(self.path, f'{name}.pkl')
            transformer.to_pickle(transformer_path)
            steps_to_save[i] = {'name': name,
                                'type': 'transformer',
                                'is_fitted': transformer.is_fitted(),
                                'path': f'{name}.pkl'}

        if self.is_prediction_pipeline():
            predictor_path = os.path.join(self.path, "model")
            self.steps[-1][1].save(predictor_path)
            steps_to_save[len(self.steps) - 1] = {'name': self.steps[-1][0],
                                                  'type': 'predictor',
                                                  'model_type': self.steps[-1][1].model_type,
                                                  'is_fitted': self.steps[-1][1].is_fitted(),
                                                  'path': 'model'}
        else:
            transformer_path = os.path.join(self.path, f'{self.steps[-1][0]}.pkl')
            self.steps[-1][1].to_pickle(transformer_path)
            steps_to_save[len(self.steps) - 1] = {'name': self.steps[-1][0],
                                                  'type': 'transformer',
                                                  'is_fitted': self.steps[-1][1].is_fitted(),
                                                  'path': f'{self.steps[-1][0]}.pkl'}
        # Save config
        config_path = os.path.join(self.path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(steps_to_save, f)

        pipeline_info = {'path': self.path}
        pipeline_path = os.path.join(self.path, 'pipeline.json')
        with open(pipeline_path, 'w') as f:
            json.dump(pipeline_info, f)

    @classmethod
    def load(cls, path: str) -> 'Pipeline':
        """
        Load the pipeline from disk.
        The sequence of transformers is loaded from a config file. The transformers and predictor are loaded from
        separate files. Transformers are loaded from pickle files, while the predictor is loaded using its own load
        method.

        Parameters
        ----------
        path: str
            Path to the directory where the pipeline is saved.

        Returns
        -------
        pipeline: Pipeline
            Loaded pipeline.
        """
        # load config
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        config = sorted(config.items(), key=lambda x: x[0])

        state_path = os.path.join(path, 'pipeline.json')
        with open(state_path, 'r') as f:
            state = json.load(f)

        steps = []
        for _, step in config:
            step_name = step['name']
            step_path = os.path.join(path, step['path'])
            step_is_fitted = step['is_fitted']
            if step['type'] == 'transformer':
                transformer = Transformer.from_pickle(step_path)
                transformer._is_fitted = step_is_fitted
                steps.append((step_name, transformer))
            elif step['type'] == 'predictor':
                predictor = _get_predictor_instance(step['model_type']).load(step_path)
                predictor._is_fitted = step_is_fitted
                steps.append((step_name, predictor))
            else:
                raise ValueError(f'Unknown step type {step["type"]}.')
        instance = cls(steps=steps, path=state['path'])
        return instance
