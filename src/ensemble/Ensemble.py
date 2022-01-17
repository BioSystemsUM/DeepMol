from abc import ABC
from typing import List

from enumerators import EnsembleClassificationEnumerators
from models.Models import Model


class Ensemble(ABC):

    def __init__(self, models: List[Model],
                 method_of_classification: EnsembleClassificationEnumerators =
                 EnsembleClassificationEnumerators.VOTING.value,
                 weights: List[float] = None):

        self.models = models
        self.method_of_classification = method_of_classification
        self.weights = weights

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value: List[Model]):
        self._models = value

    @property
    def method_of_classification(self):
        return self._models

    @method_of_classification.setter
    def method_of_classification(self, value: List[Model]):
        self._method_of_classification = value


# class BaggingEnsemble(Ensemble):

