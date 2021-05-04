'''author: Bruno Pereira
date: 28/04/2021
'''


from typing import Optional, List, Type
from copy import deepcopy
import numpy as np
from models.Models import Model
from Datasets.Datasets import Dataset
from loaders.Loaders import CSVLoader
from splitters.splitters import RandomSplitter, Splitter
from metrics.Metrics import Metric
from deepchem.models import Model as deep_model
from deepchem.models import SeqToSeq, WGAN, GATModel, GCNModel, AttentiveFPModel, LCNNModel, MultitaskIRVClassifier
from deepchem.data import NumpyDataset, DiskDataset
import deepchem as dc
#from deepchem.trans import Transformer


from utils.utils import load_from_disk, save_to_disk


def generate_sequences(epochs, train_smiles):
    """
    Function to generate the input/output pairs for SeqToSeq model
    Taken from DeepChem tutorials
    :param epochs: hyperparameter that defines the number of times  that the learning algorithm
    will work through the entire training dataset
    :param train_smiles: the ids of the samples in the dataset (smiles)
    :return: yields a pair of smile strings for epochs x len(train_smiles)
    """
    for i in range(epochs):
        for smile in train_smiles:
            yield smile, smile




class DeepChemModel(Model):
    """Wrapper class that wraps deepchem models.
    The `DeepChemModel` class provides a wrapper around deepchem
    models that allows deepchem models to be trained on `Dataset` objects
    and evaluated with the metrics in Metrics.
    """

    def __init__(self,
                 model: deep_model,
                 model_dir: Optional[str] = None,
                 **kwargs):
        """
        Parameters
        ----------
        model: deep_model
          The model instance which inherits a DeepChem `Model` Class.
        model_dir: str, optional (default None)
          If specified the model will be stored in this directory. Else, a
          temporary directory will be used.
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

        # for model in NON_WEIGHTED_MODELS:
        #     if isinstance(self.model, model):
        #         self.use_weights = False

        if 'epochs' in kwargs:
            self.epochs = kwargs['epochs']
        else:
            self.epochs = 30

    def fit(self, dataset: Dataset) -> None:
        """Fits DeepChemModel to data.
        Parameters
        ----------
        dataset: Dataset
            The `Dataset` to train this model on.
        """
        # Afraid of model.fit not recognizes the input dataset as a deepchem.data.datasets.Dataset
        

        new_dataset = NumpyDataset(
                X=dataset.X,
                y=dataset.y,
                #w = np.ones((np.shape(dataset.features)[0])),
                ids=dataset.mols)
        if isinstance(self.model, SeqToSeq):
            self.model.fit_sequences(generate_sequences(epochs=self.model.epochs, train_smiles=dataset.ids))
        elif isinstance(self.model, WGAN):
            pass
            # TODO: Wait for the implementation of iterbactches
            # self.model.fit_gan(dataset.iterbatches(5000))
        else:
            self.model.fit(new_dataset, nb_epoch=self.epochs)

    def predict(self, dataset: Dataset,
                transformers: List[dc.trans.NormalizationTransformer] = []) -> np.ndarray:
        """Makes predictions on dataset.
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
        new_dataset = NumpyDataset(
                X=dataset.X,
                y=dataset.y,
                #w = np.ones((np.shape(dataset.features)[0],self.n_tasks)),
                ids=dataset.mols)

        res =  self.model.predict(new_dataset,transformers)

        if isinstance(self.model, (GATModel,GCNModel,AttentiveFPModel,LCNNModel)):
            return res
        elif len(res.shape) == 2:
            new_res = np.squeeze(res)
        else:
            new_res = np.reshape(res,(res.shape[0],res.shape[2]))
        
        return new_res

    def predict_on_batch(self, X: Dataset) -> np.ndarray:
        """Makes predictions on batch of data.
        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on.
        """
        return super(DeepChemModel, self).predict(X)

    def save(self):
        """Saves deepchem model to disk using joblib."""
        save_to_disk(self.model, self.get_model_filename(self.model_dir))

    def reload(self):
        """Loads deepchem model from joblib file on disk."""
        self.model = load_from_disk(self.get_model_filename(self.model_dir))
 

    def cross_validate(self,
                       dataset: Dataset,
                       metric: Metric,
                       splitter: Type[Splitter],
                       transformers: List[dc.trans.NormalizationTransformer] = [],
                       folds: int = 3):
        #TODO: add option to choose between splitters (later, for now we only have random)
        #splitter = RandomSplitter()
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
            train_score = dummy_model.evaluate(train_ds, metric, transformers)
            train_scores.append(train_score[metric.name])
            avg_train_score += train_score[metric.name]

            print('Test Score: ')
            test_score = dummy_model.evaluate(test_ds, metric, transformers)
            test_scores.append(test_score[metric.name])
            avg_test_score += test_score[metric.name]

            if test_score[metric.name] > test_score_best_model:
                test_score_best_model = test_score[metric.name]
                train_score_best_model = train_score[metric.name]
                best_model = dummy_model


        return best_model, train_score_best_model, test_score_best_model, train_scores, test_scores, avg_train_score/folds, avg_test_score/folds


if __name__ == "__main__":
    # TODO: To test functions but need to fix erros with imports first
    ds = CSVLoader('../preprocessed_dataset.csv', 'Smiles', ['Class'], 'PubChem CID', chunk_size=1000)
    print(ds)
