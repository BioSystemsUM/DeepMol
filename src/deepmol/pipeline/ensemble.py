from typing import List, Literal, Tuple, Dict, Union
from collections import Counter

import numpy as np

from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.datasets import Dataset


class VotingPipeline:

    def __init__(self, pipelines: List[Pipeline], voting: Literal["hard", "soft"] = "hard", weights: List[float] = None):
        super().__init__()
        self.pipelines = pipelines
        self.voting = voting
        self.weights = weights
        self._validate_pipelines()

    def _validate_pipelines(self):
        if self.weights is None:
            self.weights = [1] * len(self.pipelines)
        else:
            # Normalize weights
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        assert len(self.weights) == len(self.pipelines), "Number of weights must be equal to number of pipelines"
        assert self.voting in ["hard", "soft"], "Voting must be either hard or soft"
        for pipeline in self.pipelines:
            assert pipeline.is_prediction_pipeline(), "All pipelines must be prediction pipelines"
        # TODO: more verifications here?
        return self

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset = None) -> 'VotingPipeline':
        for pipeline in self.pipelines:
            pipeline.fit(train_dataset, validation_dataset)
        return self

    def is_fitted(self):
        for pipeline in self.pipelines:
            if not pipeline.is_fitted():
                return False
        return True

    def _voting(self, predictions, proba=False):
        if self.voting == "hard":
            return self._hard_voting(predictions, proba)
        else:
            return self._soft_voting(predictions, proba)

    @staticmethod
    def _hard_voting(predictions, proba=False):
        if not proba:
            return [Counter(column).most_common(1)[0][0] for column in zip(*predictions)]
        else:
            # convert probabilities of each pipeline to binary predictions
            binary_predictions = []
            for pipeline_predictions in predictions:
                binary_predictions.append([1 if prediction >= 0.5 else 0 for prediction in pipeline_predictions])
            return [Counter(column).most_common(1)[0][0] for column in zip(*binary_predictions)]

    def _soft_voting(self, predictions, proba=False):
        # Calculate the weighted average of predicted probabilities
        soft_votes = np.average(predictions, axis=0, weights=self.weights)
        # if binary predictions, transform to 2D array
        if soft_votes.ndim == 1:
            soft_votes = np.array([1 - soft_votes, soft_votes]).T
        # Choose the class with the highest probability for each instance
        if not proba:
            soft_predictions = np.argmax(soft_votes, axis=1)
        else:
            soft_predictions = [soft_votes[1] for soft_votes in soft_votes]
        return soft_predictions

    def predict(self, dataset):
        predictions = []
        for pipeline in self.pipelines:
            predictions.append(pipeline.predict(dataset))
        # TODO: check for different modes
        if dataset.mode == 'classification':
            return self._voting(predictions)
        else:
            return np.average(predictions, axis=0, weights=self.weights)

    def predict_proba(self, dataset):
        # TODO: raise error for regression?
        predictions = []
        for pipeline in self.pipelines:
            predictions.append(pipeline.predict_proba(dataset))
        return self._voting(predictions, proba=True)

    def evaluate(self, dataset: Dataset, metrics: List[Metric],
                 per_task_metrics: bool = False) -> Tuple[Dict, Union[None, Dict]]:
        evaluations = []
        for pipeline in self.pipelines:
            evaluations.append(pipeline.evaluate(dataset, metrics, per_task_metrics))
        # TODO: implement this
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


if __name__ == '__main__':
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from deepmol.models import SklearnModel
    from deepmol.datasets import SmilesDataset, Dataset
    from deepmol.compound_featurization import TwoDimensionDescriptors, MorganFingerprint
    X = np.array(['O(C(=O)C(NC(=O)C(N)CC(O)=O)CCCCNOC(=O)C)C', 'Clc1ccc(cc1)C[N](Cc1ccccc1)(CC(=O)Nc1c(C)cccc1C)C',
                  'COc1ccc(cc1O)C1SCc2c(CS1)cccc2', 'CC(=O)OCC12C(CCC(C2C(=O)O)(C)C)OC(=O)C23C1C(O)CC(C2)C(C3=O)C',
                  'COc1ccc(C2OCc3ccccc3O2)cc1OC(=O)Nc1ccc([N+](=O)[O-])cc1'])
    y = np.array([0, 1, 0, 1, 0])
    data = SmilesDataset(smiles=X, y=y)
    svc = SVC(probability=True)
    scv = SklearnModel(model=svc, model_dir='model')
    rf = RandomForestClassifier()
    model = SklearnModel(model=rf, model_dir='model')
    pipe1 = Pipeline(steps=[('descriptors', TwoDimensionDescriptors()), ('model', model)], path='pipe1/')
    pipe1.fit_transform(data)
    pipe2 = Pipeline(steps=[('descriptors', MorganFingerprint()), ('model', scv)], path='pipe2/')
    pipe2.fit_transform(data)
    voting = VotingPipeline(pipelines=[pipe1, pipe2], voting='soft')
    print(voting.predict(data))
    print(voting.predict_proba(data))




