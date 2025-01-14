import copy
from unittest.mock import MagicMock

import numpy as np

from deepmol.base import Transformer, Predictor


class SmilesDatasetMagicMock(MagicMock):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __deepcopy__(self, memo):
        # Create a copy of the object using copy.deepcopy
        new_obj = MagicMock()
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            setattr(new_obj, k, copy.deepcopy(v, memo))

        return new_obj


class MockTransformerMagicMock(MagicMock, Transformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        super(Transformer, self).__init__()
        self.value = kwargs.get('value', 1)

    def _fit(self, dataset):
        return self

    def _transform(self, dataset):
        dataset._X = dataset.X + self.value
        return dataset


class MockPredictorMagicMock(MagicMock, Predictor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        super(Predictor, self).__init__()
        self.seed = kwargs.get('seed', 123)

    @property
    def model_type(self) -> str:
        return 'mock'

    def _fit(self, dataset):
        return self

    def predict(self, dataset, return_invalid=False):
        predictions = np.random.RandomState(self.seed).randint(0, 2, size=dataset.X.shape[0])
        return predictions

    def predict_proba(self, dataset, return_invalid=False):
        predictions = np.random.RandomState(self.seed).randint(0, 2, size=dataset.X.shape[0])
        return np.vstack([1 - predictions, predictions]).T

    def evaluate(self, dataset, metrics, per_task_metrics):
        metrics = {metric.name: metric.compute_metric(dataset) for metric in metrics}
        if per_task_metrics:
            metrics2 = metrics
        else:
            metrics2 = {}
        return metrics, metrics2

    def save(self, path):
        pass

    def load(self, path):
        return self


class MockMetricMagicMock(MagicMock):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'mock_metric')

    def compute_metric(self, dataset):
        return 0.5, None
