import os
from typing import List, Literal, Tuple, Dict, Union
from collections import Counter

import numpy as np

from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.datasets import Dataset
from deepmol.utils.utils import normalize_labels_shape


from scipy.stats import mode


class VotingPipeline:
    """
    Pipeline that combines the predictions of multiple pipelines using voting.
    """

    def __init__(self, pipelines: List[Pipeline],
                 voting: Literal["hard", "soft"] = "hard",
                 weights: List[float] = None) -> None:
        """
        Initializes a voting pipeline.

        Parameters
        ----------
        pipelines: List[Pipeline]
            List of pipelines to be used in the voting.
        voting: Literal["hard", "soft"]
            Type of voting to be used. Either hard or soft. Only applicable for classification.
        weights: List[float]
            List of weights to be used for each pipeline. If None, all pipelines will have equal weight.
        """
        super().__init__()
        self.pipelines = pipelines
        self.voting = voting
        self.weights = weights
        self._validate_pipelines()

    def _validate_pipelines(self) -> 'VotingPipeline':
        """
        Validates the pipelines.
        It normalizes the weights and verifies that the number of weights is equal to the number of pipelines.
        It also verifies that all pipelines are prediction pipelines.
        """
        if self.weights is None:
            self.weights = np.array([1] * len(self.pipelines))
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = np.array([w / total_weight for w in self.weights])
        assert len(self.weights) == len(self.pipelines), "Number of weights must be equal to number of pipelines"
        assert self.voting in ["hard", "soft"], "Voting must be either hard or soft"
        for pipeline in self.pipelines:
            assert pipeline.is_prediction_pipeline(), "All pipelines must be prediction pipelines"
        return self

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset = None) -> 'VotingPipeline':
        """
        Fits the pipelines to the training dataset. A separate validation dataset can also be provided.

        Parameters
        ----------
        train_dataset: Dataset
            Dataset to be used for training.
        validation_dataset: Dataset
            Dataset to be used for validation.
        """
        for pipeline in self.pipelines:
            pipeline.fit(train_dataset, validation_dataset)
        return self

    def is_fitted(self) -> bool:
        """
        Returns True if all pipelines are fitted, False otherwise.

        Returns
        -------
        bool
            True if all pipelines are fitted, False otherwise.
        """
        for pipeline in self.pipelines:
            if not pipeline.is_fitted():
                return False
        return True

    def _voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Performs the voting. It can be either hard or soft.

        Parameters
        ----------
        predictions: List[np.ndarray]
            List of predictions from each pipeline.

        Returns
        -------
        np.ndarray
            List of predictions by voting.
        """
        if self.voting == "hard":
            return self._hard_voting(predictions)
        else:
            return self._soft_voting(predictions)

    def _hard_voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Performs weighted hard voting.

        Parameters
        ----------
        predictions: List[Union[List[int], List[float]]]
            List of predictions from each pipeline.

        Returns
        -------
        np.ndarray
            Array of predictions by hard voting.
        """
        # predictions = np.array(predictions, dtype=np.float64)
        # if len(predictions.shape) > 2 and predictions.shape[2] == 1:
        #     predictions = predictions.reshape((predictions.shape[0], predictions.shape[1]))

        # binary_predictions = (predictions >= 0.5).astype(int)
        # weights = np.array([int(w) for w in self.weights * 100])
        # # Repeat predictions by weights
        # weighted_predictions = np.repeat(binary_predictions, weights.astype(int), axis=0)

        # # Perform majority voting for each sample
        # final_predictions = np.apply_along_axis(
        # lambda col: Counter(col).most_common(1)[0][0], axis=0, arr=weighted_predictions.T
        # )

        # Ensure the input is a numpy array
        predictions = np.array(predictions)

        # Convert probabilistic scores to binary predictions (0 or 1), preserving NaNs
        def to_binary(values):
            binary_values = np.where(~np.isnan(values), (values >= 0.5).astype(int), np.nan)
            return binary_values

        predictions = np.apply_along_axis(to_binary, 0, predictions)

        # Check if the input is 2D or 3D
        if predictions.ndim not in (2, 3):
            raise ValueError("Input array must be either 2D or 3D.")

        # Helper function to calculate majority vote for a 1D array, prioritizing NaNs
        def vote_with_nan_handling(values):
            if np.isnan(values).any():
                return np.nan
            else:
                values = np.array(values, dtype=int)
                counts = np.bincount(values)
    
                return np.argmax(counts) 

        # If 2D, process directly
        if predictions.ndim == 2:
            # Transpose for easier iteration over samples
            transposed = predictions.T
            final_predictions = np.apply_along_axis(vote_with_nan_handling, axis=1, arr=transposed)

        # If 3D, process each task separately
        elif predictions.ndim == 3:
            n_tasks = predictions.shape[1]
            result = []
            for task_idx in range(n_tasks):
                transposed = predictions[:, task_idx, :].T
                task_votes = np.apply_along_axis(vote_with_nan_handling, axis=1, arr=transposed)
                result.append(task_votes)
            final_predictions = np.array(result)

        return np.array(final_predictions)

    def _soft_voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Performs weighted soft voting.

        Parameters
        ----------
        predictions: List[np.ndarray]
            List of predictions from each pipeline.

        Returns
        -------
        np.ndarray
            Array of predictions by soft voting.
        """
        # transform probabilities to 0 and 1 probabilities, i.e.,
        # [[0.1, 0.9], [0.8, 0.2]] -> [[[0.9, 0.1], [0.1, 0.9]], [[0.2, 0.8], [0.8, 0.2]]]
        preds = []
        for pipeline_predictions in predictions:
            preds_ = []
            for prob in pipeline_predictions:
                if isinstance(prob, float) or isinstance(prob, int) or isinstance(prob, np.float_) or isinstance(prob, np.int_):
                    preds_.append([1 - prob, prob])
                elif isinstance(prob, np.ndarray):
                    preds_.append(list(prob))
            preds.append(preds_)
        # Calculate the weighted average of predicted probabilities
        soft_votes = np.average(preds, axis=0, weights=self.weights)
        nan_mask = np.isnan(soft_votes).any(axis=1)
    
        # Compute argmax for rows without NaN
        result = np.full(soft_votes.shape[0], np.nan)  # Initialize all values as NaN
        valid_rows = ~nan_mask
        result[valid_rows] = np.argmax(soft_votes[valid_rows], axis=1)
        # TODO: should this return classes or probabilities?
        return result
    
    def _deal_with_nan(self, return_invalid, final_predictions):

        if not return_invalid:
            if len(final_predictions.shape) > 1:
                predictions_idx = ~np.isnan(final_predictions).any(axis=1)
            else:
                predictions_idx = ~np.isnan(final_predictions)
            final_predictions = final_predictions[predictions_idx]

        return final_predictions
    
    def predict(self, dataset: Dataset, return_invalid: bool = False) -> np.ndarray:
        """
        Makes predictions for the given dataset using the voting pipeline.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used for prediction.
        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        np.ndarray
            Array of predictions.
        """
        predictions = []
        if dataset.mode == 'classification':
            for pipeline in self.pipelines:
                predictions.append(pipeline.predict_proba(dataset, return_invalid=True))

            predictions = self._voting(predictions)

            predictions = self._deal_with_nan(return_invalid, predictions)

            return predictions
        
        elif dataset.mode == 'regression':
            for pipeline in self.pipelines:
                predictions.append(pipeline.predict_proba(dataset, return_invalid=True))

            predictions = np.average(predictions, axis=0, weights=self.weights)

            predictions = self._deal_with_nan(return_invalid, predictions)

            return predictions
        else:
            for pipeline in self.pipelines:
                predictions.append(pipeline.predict_proba(dataset, return_invalid=True))

            final_predictions = np.empty(predictions[0].shape)

            # Vectorized operations for regression tasks
            regression_indices = [i for i, mode in enumerate(dataset.mode) if mode == 'regression']
            if regression_indices:
                regression_predictions = np.average([prediction[:, regression_indices] for prediction in predictions],
                                                    axis=0, weights=self.weights)
                final_predictions[:, regression_indices] = regression_predictions

            # Loop for classification tasks - potential for further optimization depending on the _voting implementation
            for i, mode in enumerate(dataset.mode):
                if mode == 'classification':
                    predictions_i = [prediction[:, i] for prediction in predictions]
                    final_predictions[:, i] = self._voting(predictions_i)

            final_predictions = self._deal_with_nan(return_invalid, final_predictions)

            return final_predictions
        
    def predict_proba(self, dataset: Dataset, return_invalid: bool = False) -> np.ndarray:
        """
        Makes predictions for the given dataset using the voting pipeline.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used for prediction.
        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        np.ndarray
            Array of predictions.
        """
        predictions = []
        if dataset.mode == 'classification':
            for pipeline in self.pipelines:
                predictions.append(pipeline.predict_proba(dataset, return_invalid=True))

            predictions = np.average(predictions, axis=0, weights=self.weights)

            predictions = self._deal_with_nan(return_invalid, predictions)

            return predictions

        elif dataset.mode == 'regression':
            for pipeline in self.pipelines:
                predictions.append(pipeline.predict_proba(dataset, return_invalid=True))

            predictions = np.average(predictions, axis=0, weights=self.weights)

            predictions = self._deal_with_nan(return_invalid, predictions)

            return predictions
        
        else:
            for pipeline in self.pipelines:
                predictions.append(pipeline.predict_proba(dataset, return_invalid=True))

            final_predictions = np.empty(predictions[0].shape)

            # Vectorized operations for regression tasks
            regression_indices = [i for i, mode in enumerate(dataset.mode) if mode == 'regression']
            if regression_indices:
                regression_predictions = np.average([prediction[:, regression_indices] for prediction in predictions],
                                                    axis=0, weights=self.weights)
                final_predictions[:, regression_indices] = regression_predictions

            # Loop for classification tasks - potential for further optimization depending on the _voting implementation
            for i, mode in enumerate(dataset.mode):
                if mode == 'classification':
                    predictions_i = [prediction[:, i] for prediction in predictions]
                    final_predictions[:, i] = np.average(predictions_i, axis=0, weights=self.weights)

            final_predictions = self._deal_with_nan(return_invalid, final_predictions)

            return final_predictions

    def evaluate(self, dataset: Dataset, metrics: List[Metric],
                 per_task_metrics: bool = False) -> Tuple[Dict, Union[None, Dict]]:
        """
        Evaluates the voting pipeline using the given metrics.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be used for evaluation.
        metrics: List[Metric]
            List of metrics to be used.
        per_task_metrics: bool
            If true, returns the metrics for each task separately.

        Returns
        -------
        Tuple[Dict, Union[None, Dict]]
            Tuple containing the multitask scores and the scores for each task separately.
        """
        n_tasks = dataset.n_tasks
        y = dataset.y
        y_pred = self.predict(dataset)
        if not y.shape == np.array(y_pred).shape:
            y_pred = normalize_labels_shape(y_pred, n_tasks)

        multitask_scores = {}
        all_task_scores = {}

        # Compute multitask metrics
        for metric in metrics:
            results = metric.compute_metric(y, y_pred, per_task_metrics=per_task_metrics, n_tasks=n_tasks)
            if per_task_metrics:
                multitask_scores[metric.name], computed_metrics = results
                all_task_scores[metric.name] = computed_metrics
            else:
                multitask_scores[metric.name], all_task_scores = results

        if not per_task_metrics:
            return multitask_scores, {}
        else:
            return multitask_scores, all_task_scores

    def save(self, path: str) -> 'VotingPipeline':
        """
        Saves the voting pipeline.

        Parameters
        ----------
        path: str
            Path where the voting pipeline will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        for pipeline in self.pipelines:
            pipeline.path = os.path.join(path, os.path.basename(os.path.normpath(pipeline.path)))
            pipeline.save()
        # save json with voting and weights info
        with open(os.path.join(path, 'voting_pipeline.json'), 'w') as f:
            f.write(f'{{"voting": "{self.voting}", "weights": {self.weights.tolist()}}}')
        return self

    @classmethod
    def load(cls, path: str) -> 'VotingPipeline':
        """
        Loads a voting pipeline from the specified path.

        Parameters
        ----------
        path: str
            Path where the voting pipeline is saved.

        Returns
        -------
        VotingPipeline
            Loaded voting pipeline.
        """
        pipelines = []
        for pipeline_path in os.listdir(path):
            if pipeline_path != 'voting_pipeline.json':
                pp = os.path.join(path, pipeline_path)
                pipelines.append(Pipeline.load(pp))
        # load json with voting and weights info
        with open(os.path.join(path, 'voting_pipeline.json'), 'r') as f:
            voting_info = f.read()
        voting_info = eval(voting_info)
        return cls(pipelines=pipelines, voting=voting_info['voting'], weights=voting_info['weights'])
