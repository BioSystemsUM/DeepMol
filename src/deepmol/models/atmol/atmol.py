import os
import pickle
from typing import Union
from torch import nn
import pytorch_lightning as pl
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from deepmol.datasets.datasets import Dataset
from deepmol.models._utils import _return_invalid, get_prediction_from_proba
from deepmol.models.atmol.encoder_gat import GATNet
from deepmol.models.atmol.model_gat_pre import GATCon
from deepmol.models.atmol.nt_xent import NT_Xent
from deepmol.models.atmol.utils_gat_pretrain import AtmolTorchDataset
from deepmol.models.models import Model

from torch.optim import Adam

from torch.nn import BCELoss

from deepmol.utils.utils import normalize_labels_shape

class AtMolLightning(pl.LightningModule, Model):

    def __init__(self, temperature=0.1, n_output=128, lr=0.0001, weight_decay=1e-7, 
                 batch_size=128, mode = "masked_learning", optimizer = Adam, loss = BCELoss,
                 prediction_head_layers: list = [512, 256, 128], metric = None,
                 **trainer_args):
        
        super().__init__()
        self.temperature = temperature

        self.loss = loss()
        self.optimizer = optimizer
        
        self.encoder1 = GATNet(output_dim=n_output)
        self.encoder2 = GATNet(output_dim=n_output)

        self.mode = mode
        self.trainer_args = trainer_args
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_output = n_output

        self.metric = None

        self.model = GATCon(n_output, encoder1=self.encoder1, encoder2=self.encoder2)

        self.prediction_head_layers = prediction_head_layers
        
        self.head = nn.Sequential(
                nn.Linear(self.n_output, prediction_head_layers[0]),
                nn.ReLU(),
                nn.BatchNorm1d(prediction_head_layers[0]))

        if len(prediction_head_layers) > 1:
            for i, layer in enumerate(prediction_head_layers[1:]):
                i += 1
                self.head.add_module(f"layer_{i}", nn.Linear(prediction_head_layers[i-1], layer))
                self.head.add_module(f"relu_{i}", nn.ReLU())
                self.head.add_module(f"batch_norm_{i}", nn.BatchNorm1d(layer))

    def _forward_fine_tuning(self, x):
        x, edge_index, batch, x_size, edge_size = x.x, x.edge_index,x.batch, x.x_size, x.edge_size

        x1, weight1 = self.encoder1(x, edge_index, batch)

        x = self.head(x1)
        if isinstance(self.dataset_mode, list):
            if self.dataset_mode[0] == "classification":
                x = nn.Sigmoid()(x)
        elif self.dataset_mode == "classification":
            x = nn.Sigmoid()(x)

        return x
        
    def forward(self, x):
        if self.mode == "masked_learning":
            return self.model(x)
        else:
            return self._forward_fine_tuning(x)
    
    def _fit(self, dataset: Union[Dataset, AtmolTorchDataset], validation_dataset: Union[Dataset, AtmolTorchDataset] = None):
        
        self.dataset_mode = dataset.mode

        if isinstance(self.dataset_mode, list):
            self.head.add_module("last_layer", nn.Linear(self.prediction_head_layers[-1], len(self.dataset_mode)))
        else:
            self.head.add_module("last_layer", nn.Linear(self.prediction_head_layers[-1], 1))

        if not isinstance(dataset, AtmolTorchDataset):
            dataset = AtmolTorchDataset(dataset).featurize()

        dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if validation_dataset is not None:
            if not isinstance(validation_dataset, AtmolTorchDataset):
                validation_dataset = AtmolTorchDataset(validation_dataset).featurize()
                validation_dataset = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
            

        model = AtMolLightning(self.temperature, self.n_output)
        self.trainer = pl.Trainer(**self.trainer_args)

        self.trainer.fit(model, dataset, validation_dataset)

    def training_step(self, batch, batch_idx):
        if self.mode == "masked_learning":
            _, out_1, _, out_2, _, _ = self(batch)
            criterion = NT_Xent(out_1.shape[0], self.temperature, 1)
            loss = criterion(out_1, out_2)
        else:
            y_pred = self(batch)
            loss = self.loss(y_pred, batch.y)

        self.log('train_loss', loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        
        if self.metric != None:
            score = self.metric(y_pred, batch.y)
            self.log(f'train_{self.metric.__class__.__name__}', score, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == "masked_learning":
            _, out_1, _, out_2, _, _ = self(batch)
            criterion = NT_Xent(out_1.shape[0], self.temperature, 1)
            loss = criterion(out_1, out_2)
        else:
            y_pred = self(batch)
            loss = self.loss(y_pred, batch.y)

        self.log('val_loss', loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        
        if self.metric != None:
            score = self.metric(y_pred, batch.y)
            self.log(f'val_{self.metric.__class__.__name__}', score, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        outputs = self(batch)
        return outputs

    def predict(self, dataset: Dataset, return_invalid: bool = False) -> np.ndarray:
        """
        Makes predictions on dataset.

        Parameters
        ----------
        dataset: Dataset
          Dataset to make prediction on.
        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        np.ndarray
          The value is a return value of `predict_proba` or `predict` method of the scikit-learn model. If the
          scikit-learn model has both methods, the value is always a return value of `predict_proba`.
        """
        predictions = self.predict_proba(dataset)

        y_pred_rounded = get_prediction_from_proba(dataset, predictions)

        if return_invalid:
            y_pred_rounded = _return_invalid(dataset, y_pred_rounded)
        return y_pred_rounded

    def predict_proba(self, dataset: Dataset, return_invalid: bool = False) -> np.ndarray:
        """
        Makes predictions on dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to make prediction on.

        return_invalid: bool
            Return invalid entries with NaN

        Returns
        -------
        np.ndarray
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        predictions = self.trainer.predict(self, dataloader)

        if type(predictions[0]) == tuple:
            len_tuple = len(predictions[0])
            new_predictions = [None] * len_tuple
            for i in range(len_tuple):
                new_predictions[i] = [prediction[i] for prediction in predictions]
                new_predictions[i] = torch.cat(new_predictions[i]).detach().cpu().numpy()
            predictions = new_predictions
        else:
            predictions = torch.cat(predictions)
            # convert to numpy array
            predictions = predictions.detach().cpu().numpy()

        if len(predictions.shape) > 1:
            if predictions.shape != (len(dataset.data), dataset.n_tasks):
                predictions = normalize_labels_shape(predictions, dataset.n_tasks)

        if return_invalid:
            predictions = _return_invalid(dataset, predictions)

        return predictions
    

    def configure_optimizers(self):
        if self.mode == "masked_learning":
            return self.optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            return self.optimizer(self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def save(self, file_path: str = None):

        if file_path is None:
            pl_model_path = os.path.join(self.model_dir, "model.ckpt")
        else:
            os.makedirs(file_path, exist_ok=True)
            pl_model_path = os.path.join(file_path, "model.ckpt")

        self.trainer.save_checkpoint(pl_model_path)

        # with open(os.path.join(file_path, "trainer.pk"), "wb") as b:
        #     pickle.dump(self.trainer, b, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(file_path, "model.pk"), "wb") as b:
            pickle.dump(self, b, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, folder_path: str) -> 'AtMolLightning':

        with open(os.path.join(folder_path, "model.pk"), "rb") as b:
            new_model = pickle.load(b)

        # with open(os.path.join(folder_path, "trainer.pk"), "rb") as b:
        #     trainer = pickle.load(b)

        # new_model.trainer = trainer

        return new_model