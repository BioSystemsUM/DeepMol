import os
import pickle
import shutil
import tempfile
from typing import List, Sequence, Union
import numpy as np
import torch

from deepmol.base import Predictor
from deepmol.evaluator import Evaluator
from deepmol.metrics.metrics import Metric
from deepmol.datasets import Dataset

try:
    from deepchem.models.torch_models import TorchModel
    from deepchem.models import SeqToSeq, WGAN
    from deepchem.models import Model as BaseDeepChemModel
    from deepchem.data import NumpyDataset
    import deepchem as dc
except ImportError:
    pass

from deepmol.models._utils import _get_splitter, _return_invalid, save_to_disk, load_from_disk, get_prediction_from_proba
from deepmol.splitters.splitters import Splitter
from deepmol.utils.utils import normalize_labels_shape


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


class DeepChemModel(BaseDeepChemModel, Predictor):
    """
    Wrapper class that wraps deepchem models.
    The `DeepChemModel` class provides a wrapper around deepchem models that allows deepchem models to be trained on
    `Dataset` objects and evaluated with the metrics in Metrics.
    """

    model: BaseDeepChemModel

    def __init__(self,
                 model: BaseDeepChemModel,
                 model_dir: str = None,
                 custom_objects: dict = None,
                 **kwargs):
        """
        Initializes a DeepChemModel.

        Parameters
        ----------
        model: BaseDeepChemModel
          The model instance which inherits a DeepChem `Model` Class.
        model_dir: str, optional (default None)
          If specified the model will be stored in this directory. Else, a temporary directory will be used.
        custom_objects: dict, optional (default None)
            Dictionary of custom objects to be passed to the model.
        kwargs:
          additional arguments to be passed to the model.
        """

        if 'model_dir' in kwargs:
            model_dir = kwargs.pop('model_dir')

        if model_dir is None:
            model_dir = tempfile.mkdtemp()

        if 'epochs' in kwargs:
            self.epochs = kwargs.pop('epochs')
        else:
            self.epochs = 30

        assert isinstance(model, type), f"Model must be a class not an instance. Got {type(model)}"

        self.model_instance = model
        model = model(**kwargs)
        self._define_model_mode_in_multitask_models(model)

        super().__init__(model=model, model_dir=model_dir, epochs=self.epochs, **kwargs)
        super(Predictor, self).__init__()
        self._model_dir = model_dir

        if 'use_weights' in kwargs:
            self.use_weights = kwargs['use_weights']
        else:
            self.use_weights = True

        if 'n_tasks' in kwargs:
            self.n_tasks = kwargs['n_tasks']
        else:
            self.n_tasks = 1

        self.custom_objects = custom_objects

        self.deepchem_model_parameters = kwargs

        self.parameters_to_save = {
            'use_weights': self.use_weights,
            'epochs': self.epochs,
            'model_instance': self.model_instance
        }

    def _define_model_mode_in_multitask_models(self, model):
        """
        Defines the model mode in multitask models.
        """
        if str(model.__class__.__name__) in ["MultitaskClassifier"]:
            model.mode = 'classification'
            model.model.mode = 'classification'

        elif str(model.__class__.__name__) in [
                                           "MultitaskIRVClassifier",
                                           "ProgressiveMultitaskClassifier",
                                           "RobustMultitaskClassifier",
                                           "ScScoreModel"
                                           ]:
                                      
            model.model.mode = 'classification'
        
        elif str(model.__class__.__name__) in ["ProgressiveMultitaskRegressor", 
                                           "RobustMultitaskRegressor",
                                           "MATModel"]:
            model.model.mode = 'regression'

        elif str(model.__class__.__name__) in ["MultitaskRegressor"]:
            model.mode = 'regression'
            model.model.mode = 'regression'

        else:
            pass


    @property
    def model_type(self):
        """
        Returns the type of the model.
        """
        return 'deepchem'

    def fit(self, dataset: Dataset):
        """
        Fits the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        """
        Predictor.fit(self, dataset)

    def fit_on_batch(self, X: Sequence, y: Sequence, w: Sequence):
        """
        Fits the model on a batch of data.

        Parameters
        ----------
        X: Sequence
            The input data.
        y: Sequence
            The output data.
        w: Sequence
            The weights for the data.
        """

    def get_task_type(self) -> str:
        """
        Returns the task type of the model.

        Returns
        -------
        str
            The task type of the model.
        """

    def get_num_tasks(self) -> int:
        """
        Returns the number of tasks of the model.

        Returns
        -------
        int
            The number of tasks of the model.
        """

    def _fit(self, dataset: Dataset) -> None:
        """
        Fits DeepChemModel to data.

        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        """
        # TODO: better way to validate model.mode and dataset.mode
        if not isinstance(dataset.mode, list):
            if hasattr(self.model, 'mode'):
                model_mode = self.model.mode
                if model_mode != dataset.mode:
                    raise ValueError(f"The model mode and the dataset mode must be the same. "
                                     f"Got model mode: {model_mode} and dataset mode: {dataset.mode}")
            else:
                model_mode = self.model.model.mode
                if model_mode != dataset.mode:
                    raise ValueError(f"The model mode and the dataset mode must be the same. "
                                     f"Got model mode: {model_mode} and dataset mode: {dataset.mode}")
        else:
            model_mode = dataset.mode
        # Afraid of model.fit not recognizes the input dataset as a deepchem.data.datasets.Dataset
        if isinstance(self.model, TorchModel) and model_mode == 'regression':
            y = np.expand_dims(dataset.y, axis=-1)  # need to do this so that the loss is calculated correctly
        else:
            y = dataset.y
        new_dataset = NumpyDataset(X=dataset.X, y=y, ids=dataset.ids, n_tasks=dataset.n_tasks)

        if isinstance(self.model, SeqToSeq):
            self.model.fit_sequences(generate_sequences(epochs=self.model.epochs, train_smiles=dataset.smiles))
        elif isinstance(self.model, WGAN):
            pass
            # TODO: Wait for the implementation of iterbactches
            # self.model.fit_gan(dataset.iterbatches(5000))
        else:
            self.model.fit(new_dataset, nb_epoch=self.epochs)

    def predict(self,
                dataset: Dataset,
                transformers: List[dc.trans.NormalizationTransformer] = None,
                return_invalid: bool = False
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

        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        np.ndarray
            The value is a return value of `predict` method of the DeepChem model.
        """

        predictions = self.predict_proba(dataset, transformers)
        y_pred_rounded = get_prediction_from_proba(dataset, predictions)

        if return_invalid:
            y_pred_rounded = _return_invalid(dataset, y_pred_rounded)

        return y_pred_rounded

    def predict_proba(self,
                      dataset: Dataset,
                      transformers: List[dc.trans.NormalizationTransformer] = None,
                      return_invalid: bool = False
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

        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        np.ndarray
            The value is a return value of `predict` method of the DeepChem model.
        """
        if transformers is None:
            transformers = []
        new_dataset = NumpyDataset(X=dataset.X, y=dataset.y, ids=dataset.ids, n_tasks=dataset.n_tasks)

        res = self.model.predict(new_dataset, transformers)

        if isinstance(self.model, TorchModel) and self.model.model.mode == 'classification':
            
            if return_invalid:
                res = _return_invalid(dataset, res)

            return res
        else:
            new_res = np.squeeze(
                res)  

        if dataset.mode is not None:
            if not isinstance(dataset.mode, str):
                n_tasks = len(dataset.mode)
                if new_res.shape != (len(dataset.mols), n_tasks):
                    new_res = normalize_labels_shape(new_res, n_tasks)
        else:
            error_message = """Mode is not defined, please define it when creating the dataset
                    Example with CSVLoader: CSVLoader(path_to_csv, smiles_field='smiles', mode='classification'),
                    Example with SmilesDataset: SmilesDataset(smiles, mode='classification')
                    Example with SmilesDataset for multitask classification with 10 tasks: 
                    SmilesDataset(smiles, mode=['classification']*10)"""
            raise ValueError(error_message)

        if len(new_res.shape) > 1:
            if new_res.shape[1] == len(dataset.mols) and new_res.shape[0] == dataset.n_tasks:
                new_res = new_res.T

        if return_invalid:
            new_res = _return_invalid(dataset, new_res)

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

    def save(self, folder_path: str = None):
        """
        Saves deepchem model to disk.

        Parameters
        ----------
        folder_path: str
            Path to the file where the model will be stored.
        """
        if folder_path is None:
            if self.model_dir is None:
                raise ValueError("Please specify folder_path or model_dir")
            folder_path = self.model_dir
        else:
            self.model_dir = folder_path
            os.makedirs(folder_path, exist_ok=True)

        if os.path.exists(os.path.join(folder_path, "model")):
            shutil.rmtree(os.path.join(folder_path, "model"))

        shutil.copytree(self.model.model_dir, os.path.join(folder_path, "model"))

        save_to_disk(self.parameters_to_save, os.path.join(folder_path, "model_parameters.pkl"))

        save_to_disk(self.deepchem_model_parameters, os.path.join(folder_path, "deepchem_model_parameters.pkl"))

        if self.custom_objects is not None:
            with open(os.path.join(folder_path, 'custom_objects.pkl'), 'wb') as file:
                pickle.dump(self.custom_objects, file)


    @classmethod
    def load(cls, folder_path: str, **kwargs):
        """
        Loads deepchem model from disk.

        Parameters
        ----------
        folder_path: str
            Path to the file where the model is stored.
        kwargs: Dict
            Additional parameters.
            custom_objects: Dict
                Dictionary of custom objects to be passed to `tensorflow.keras.utils.custom_object_scope`.
        """

        deepchem_model_parameters = load_from_disk(os.path.join(folder_path, "deepchem_model_parameters.pkl"))
        model_parameters = load_from_disk(os.path.join(folder_path, "model_parameters.pkl"))

        model = model_parameters.pop('model_instance')
        model_parameters.update(deepchem_model_parameters)

        deepchem_model = cls(model=model, 
                             model_dir=os.path.join(folder_path, "model"), **model_parameters)
        
        if not torch.cuda.is_available():
            deepchem_model.model.device = "cpu"
        try:
            deepchem_model.model.restore(model_dir=os.path.join(folder_path, "model"))
        except ValueError:
            print("The model was not restored. The model was probably not trained.")

        return deepchem_model

    def cross_validate(self,
                       dataset: Dataset,
                       metric: Metric,
                       splitter: Splitter = None,
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
            Splitter to use for cross validation.
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
        if splitter is None:
            splitter = _get_splitter(dataset)
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

            dummy_model = DeepChemModel(self.model.__class__, **self.deepchem_model_parameters, **self.parameters_to_save)

            # TODO: isto está testado ? estes transformers nao é um boleano
            train_score = dummy_model.evaluate(train_ds, [metric], transformers)
            train_scores.append(train_score[0][metric.name])
            avg_train_score += train_score[0][metric.name]

            test_score = dummy_model.evaluate(test_ds, [metric], transformers)
            test_scores.append(test_score[0][metric.name])
            avg_test_score += test_score[0][metric.name]

            if test_score[0][metric.name] > test_score_best_model:
                test_score_best_model = test_score[0][metric.name]
                train_score_best_model = train_score[0][metric.name]
                best_model = dummy_model

        return best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, avg_train_score / folds, avg_test_score / folds

    def evaluate(self,
                 dataset: Dataset,
                 metrics: List[Metric],
                 per_task_metrics: bool = False):
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

        Returns
        -------
        Tuple[Dict, Dict]
            multitask_scores: dict
                Dictionary mapping names of metrics to metric scores.
            all_task_scores: dict
                If `per_task_metrics == True`, then returns a second dictionary of scores for each task separately.
        """
        evaluator = Evaluator(self, dataset)
        return evaluator.compute_model_performance(metrics, per_task_metrics=per_task_metrics)