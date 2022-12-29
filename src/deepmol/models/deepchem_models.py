from typing import List, Sequence, Union
import numpy as np

from deepmol.evaluator import Evaluator
from deepmol.metrics.metrics import Metric
from deepmol.datasets import Dataset
from deepchem.models.torch_models import TorchModel
from deepchem.models import SeqToSeq, WGAN
from deepchem.models import Model as BaseDeepChemModel
from deepchem.data import NumpyDataset
import deepchem as dc

from deepmol.splitters.splitters import Splitter
from deepmol.utils.utils import load_from_disk, save_to_disk


def generate_sequences(epochs: int, train_smiles: List[Union[str, int]]):
    """
    Function to generate the input/output pairs for SeqToSeq model.
    Taken from DeepChem tutorials.

    Parameters
    ----------
    epochs : int
        Number of epochs to train the model.
    train_smiles : List[str]
        The ids of the samples in the dataset (smiles)

    Returns
    -------
    yields a pair of smile strings for epochs x len(train_smiles)
    """
    for i in range(epochs):
        for smile in train_smiles:
            yield smile, smile


class DeepChemModel(BaseDeepChemModel):
    """
    Wrapper class that wraps deepchem models.
    The `DeepChemModel` class provides a wrapper around deepchem models that allows deepchem models to be trained on
    `Dataset` objects and evaluated with the metrics in Metrics.
    """

    def __init__(self,
                 model: BaseDeepChemModel,
                 model_dir: str = None,
                 **kwargs):
        """
        Initializes a DeepChemModel.

        Parameters
        ----------
        model: BaseDeepChemModel
          The model instance which inherits a DeepChem `Model` Class.
        model_dir: str, optional (default None)
          If specified the model will be stored in this directory. Else, a temporary directory will be used.
        kwargs: dict
          kwargs['use_weights'] is a bool which determines if we pass weights into
          self.model.fit().
        """
        if 'model_instance' in kwargs:
            self.model_instance = kwargs['model_instance']
            if model is not None:
                raise ValueError("Can not use both model and model_instance argument at the same time.")

            model = self.model_instance

        super(DeepChemModel, self).__init__(model, model_dir, **kwargs)
        if 'use_weights' in kwargs:
            self.use_weights = kwargs['use_weights']
        else:
            self.use_weights = True

        if 'n_tasks' in kwargs:
            self.n_tasks = kwargs['n_tasks']
        else:
            self.n_tasks = 1

        if 'epochs' in kwargs:
            self.epochs = kwargs['epochs']
        else:
            self.epochs = 30

    def fit_on_batch(self, X: Sequence, y: Sequence, w: Sequence):
        raise NotImplementedError

    def get_task_type(self) -> str:
        raise NotImplementedError

    def get_num_tasks(self) -> int:
        raise NotImplementedError

    def fit(self, dataset: Dataset) -> None:
        """Fits DeepChemModel to data.
        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        """
        # Afraid of model.fit not recognizes the input dataset as a deepchem.data.datasets.Dataset
        if isinstance(self.model, TorchModel) and self.model.model.mode == 'regression':
            y = np.expand_dims(dataset.y, axis=-1)  # need to do this so that the loss is calculated correctly
            new_dataset = NumpyDataset(
                X=dataset.X,
                y=y,
                ids=dataset.mols)
        else:
            new_dataset = NumpyDataset(
                X=dataset.X,
                y=dataset.y,
                # w = np.ones((np.shape(dataset.features)[0])),
                ids=dataset.mols)
        if isinstance(self.model, SeqToSeq):
            self.model.fit_sequences(generate_sequences(epochs=self.model.epochs, train_smiles=dataset.ids))
        elif isinstance(self.model, WGAN):
            pass
            # TODO: Wait for the implementation of iterbactches
            # self.model.fit_gan(dataset.iterbatches(5000))
        else:
            self.model.fit(new_dataset, nb_epoch=self.epochs)

    def predict(self,
                dataset: Dataset,
                transformers: List[dc.trans.NormalizationTransformer] = None
                ) -> np.ndarray:
        """
        Makes predictions on dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on.

        transformers: List[Transformer]
            Transformers that the input data has been transformed by. The output
            is passed through these transformers to undo the transformations.

        Returns
        -------
        np.ndarray
            The value is a return value of `predict` method of the DeepChem model.
        """
        if transformers is None:
            transformers = []
        new_dataset = NumpyDataset(
            X=dataset.X,
            y=dataset.y,
            # w = np.ones((np.shape(dataset.features)[0],self.n_tasks)),
            ids=dataset.mols)

        res = self.model.predict(new_dataset, transformers)

        # if isinstance(self.model, (GATModel,GCNModel,AttentiveFPModel,LCNNModel)):
        #     return res
        # elif len(res.shape) == 2:
        #     new_res = np.squeeze(res)
        # else:
        #     new_res = np.reshape(res,(res.shape[0],res.shape[2]))
        if isinstance(self.model, TorchModel) and self.model.model.mode == 'classification':
            return res
        else:
            new_res = np.squeeze(
                res)  # this works for all regression models (Keras and PyTorch) and is more general than the
            # commented code above

        return new_res

    def predict_on_batch(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions on batch of data.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on.
        """
        return super(DeepChemModel, self).predict(dataset)

    def save(self):
        """
        Saves deepchem model to disk using joblib.
        """
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """
        Loads deepchem model from joblib file on disk.
        """
        self.model = load_from_disk(self.get_model_filename(self.model_dir))

    def cross_validate(self,
                       dataset: Dataset,
                       metric: Metric,
                       splitter: Splitter,
                       transformers: List[dc.trans.NormalizationTransformer] = None,
                       folds: int = 3):
        """
        Cross validates the model on the specified dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to cross validate on.
        metric: Metric
            Metric to evaluate the model on.
        splitter: Splitter
            Splitter to split the dataset into train and test sets.
        transformers: List[Transformer]
            Transformers that the input data has been transformed by.
        folds: int
            Number of folds to use for cross validation.

        Returns
        -------
        Tuple[DeepChemModel, float, float, List[float], List[float], float, float]
            The first element is the best model, the second is the train score of the best model, the third is the train
            score of the best model, the fourth is the test scores of all models, the fifth is the average train scores
            of all folds and the sixth is the average test score of all folds.
        """
        # TODO: add option to choose between splitters (later, for now we only have random)
        # splitter = RandomSplitter()
        if transformers is None:
            transformers = []
        datasets = splitter.k_fold_split(dataset, folds)

        train_scores = []
        train_score_best_model = 0
        avg_train_score = 0

        test_scores = []
        test_score_best_model = 0
        avg_test_score = 0
        best_model = None
        for train_ds, test_ds in datasets:

            dummy_model = DeepChemModel(self.model)

            print('Train Score: ')
            # TODO: isto está testado ? estes transformers nao é um boleano
            train_score = dummy_model.evaluate(train_ds, [metric], transformers)
            train_scores.append(train_score[metric.name])
            avg_train_score += train_score[metric.name]

            print('Test Score: ')
            test_score = dummy_model.evaluate(test_ds, [metric], transformers)
            test_scores.append(test_score[metric.name])
            avg_test_score += test_score[metric.name]

            if test_score[metric.name] > test_score_best_model:
                test_score_best_model = test_score[metric.name]
                train_score_best_model = train_score[metric.name]
                best_model = dummy_model

        return best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, avg_train_score / folds, avg_test_score / folds

    def evaluate(self,
                 dataset: Dataset,
                 metrics: List[Metric],
                 per_task_metrics: bool = False,
                 n_classes: int = 2):
        """
        Evaluates the performance of the model on the provided dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to evaluate the model on.
        metrics: List[Metric]
            Metrics to evaluate the model on.
        per_task_metrics: bool
            If true, return computed metric for each task on multitask dataset.
        n_classes: int
            Number of classes in the dataset.

        Returns
        -------
        Tuple[Dict, Dict]
            multitask_scores: dict
                Dictionary mapping names of metrics to metric scores.
            all_task_scores: dict
                If `per_task_metrics == True`, then returns a second dictionary of scores for each task separately.
        """
        evaluator = Evaluator(self, dataset)
        return evaluator.compute_model_performance(metrics, per_task_metrics=per_task_metrics, n_classes=n_classes)
