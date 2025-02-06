from abc import abstractmethod
import os
import pickle
import tempfile

import numpy as np
import torch

from deepmol.loggers.logger import Logger
from deepmol.models._utils import _return_invalid, get_prediction_from_proba
from deepmol.models.models import Model
from deepmol.datasets import Dataset

from torch.utils.data import DataLoader

import torch.nn as nn

from pytorch_lightning import LightningModule, Trainer
from torch.optim import AdamW

from transformers import ModernBertConfig, ModernBertForMaskedLM, BertConfig, \
    BertForMaskedLM, RobertaForMaskedLM, RobertaConfig, DebertaForMaskedLM, DebertaConfig, ModernBertModel, \
    RobertaModel, BertModel, DebertaModel



from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.nn import BCELoss

from deepmol.utils.utils import normalize_labels_shape


class TransformerModelForMaskedLM(LightningModule, Model):

    def __init__(self, model, learning_rate=1e-4, batch_size=8, model_dir = None, mode: str = "masked_learning",
                 cls_token = True, optimizer = AdamW, loss = BCELoss,
                 **trainer_kwargs):
        
        # Initialize LightningModule
        LightningModule.__init__(self)
        self.save_hyperparameters()
        self.model = model
        self.trainer_kwargs = trainer_kwargs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cls_token = cls_token
        self.loss = loss()

        self.model_dir_is_temp = False

        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        else:
            model_dir = tempfile.mkdtemp()
            self.model_dir_is_temp = True

        self.mode = mode

        self._model_dir = model_dir
        self.model = model
        self.model_class = model.__class__

        self.optimizer = optimizer

        self.dataset_mode = None

        self.head = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
            )

    def _forward_fine_tuning(self, input_ids, attention_mask):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.cls_token:
            cls_output = outputs.last_hidden_state[:, 0, :]
        else:
            cls_output = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        head_output = self.head(cls_output)

        if isinstance(self.dataset_mode, list):
            if self.dataset_mode[0] == "classification":
                head_output = nn.Sigmoid()(head_output)
        elif self.dataset_mode == "classification":
            head_output = nn.Sigmoid()(head_output)

        return head_output
    
    def forward(self, input_ids, attention_mask, labels=None):

        if self.mode == "masked_learning":
            try:
                return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            except TypeError as e:
                raise TypeError(f"{e}. Make sure the mode is correctly set")
        else:
            return self._forward_fine_tuning(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        if self.mode == "masked_learning":
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        else:
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = self.loss(outputs, labels)

        self.log('train_loss', loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        if self.mode == "masked_learning":
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = outputs.loss
        else:
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = self.loss(outputs, labels)

        self.log('val_loss', val_loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        return val_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)
    
    def _fit(self, dataset: Dataset, validation_dataset: Dataset = None):

        self.dataset_mode = dataset.mode

        if isinstance(self.dataset_mode, list):
            self.head.add_module("last_layer", nn.Linear(1024, len(self.dataset_mode)))
        else:
            self.head.add_module("last_layer", nn.Linear(1024, 1))

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        callbacks = []

        if validation_dataset is not None:
            val_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
            monitor = "val_loss"
            early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=2, verbose=True)
            callbacks.append(early_stopping_callback)
        else:
            val_dataloader=None
            monitor='train_loss'

        # Create a checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath='checkpoints',
            filename='epoch-{epoch:02d}',
            save_top_k=5,
            every_n_epochs=1,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        self.trainer = Trainer(**self.trainer_kwargs,
                     callbacks=callbacks)  # Use gpus=1 if you have a GPU available
        self.trainer.fit(self, dataloader, val_dataloader)

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
            if predictions.shape != (len(dataset.mols), dataset.n_tasks):
                predictions = normalize_labels_shape(predictions, dataset.n_tasks)

        if return_invalid:
            predictions = _return_invalid(dataset, predictions)

        return predictions
    
    @abstractmethod
    def _load(config, model_path, mode):
        """_summary_

        Parameters
        ----------
        config : _type_
            _description_
        model_path : _type_
            _description_
        mode : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

    # def load_from_pretrained(self, config, path, mode: str = "classification"):

    #     self._load(config, path, mode)

    #     self.trainer = Trainer(accelerator="cpu")

    #     return self

    # def export_model(self, output_path):

    #     torch.save(self.model.state_dict(), output_path)

    def save(self, file_path: str = None):

        if file_path is None:
            model_path = os.path.join(self.model_dir, "model.pt")
            pl_model_path = os.path.join(self.model_dir, "model.ckpt")
        else:
            os.makedirs(file_path, exist_ok=True)
            model_path = os.path.join(file_path, "model.pt")
            pl_model_path = os.path.join(file_path, "model.ckpt")

        self.trainer.save_checkpoint(pl_model_path)
        torch.save(self.model.state_dict(), model_path)

        with open(os.path.join(file_path, "trainer.pk"), "wb") as b:
            pickle.dump(self.trainer, b, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(file_path, "model.pk"), "wb") as b:
            pickle.dump(self, b, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(file_path, "config.pk"), "wb") as b:
            pickle.dump(self.config, b, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, folder_path: str, mode: str = "classification") -> 'TransformerModelForMaskedLM':

        with open(os.path.join(folder_path, "model.pk"), "rb") as b:
            new_model = pickle.load(b)

        with open(os.path.join(folder_path, "config.pk"), "rb") as b:
        
            config = pickle.load(b)

        with open(os.path.join(folder_path, "trainer.pk"), "rb") as b:
        
            trainer = pickle.load(b)

        model_path = os.path.join(folder_path, "model.pt")

        new_model._load(config, model_path, mode)
        new_model.trainer = trainer


        return new_model
    

class ModernBERT(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, mode: str = "masked_learning",
                 cls_token = True, optimizer = AdamW, loss = BCELoss,
                 **trainer_kwargs):
        
        self.config = ModernBertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            pad_token_id = 0
        )

        if mode == "masked_learning":

            model = ModernBertForMaskedLM(self.config)
        
        else:

            model = ModernBertModel(self.config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, mode = mode,
                 cls_token = cls_token, optimizer = optimizer, loss = loss, **trainer_kwargs)
        
    def _load(self, config, model_path, mode):
        self.config = config

        if mode == "masked_learning":

            model = ModernBertForMaskedLM.from_pretrained(pretrained_model_name_or_path = model_path, config=config)
        
        else:

            model = ModernBertModel.from_pretrained(pretrained_model_name_or_path = model_path, config=config)

        self.model = model

class BERT(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, mode: str = "masked_learning",
                 cls_token = True, num_labels = 1, optimizer = AdamW, loss = BCELoss,
                 **trainer_kwargs):
        
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            pad_token_id = 0
        )

        if mode == "masked_learning":

            model = BertForMaskedLM(self.config)
        
        else:

            model = BertModel(self.config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, mode = mode,
                 cls_token = cls_token, optimizer = optimizer, loss = loss, **trainer_kwargs)
        
    def _load(self, config, model_path, mode):
        self.config = config

        if mode == "masked_learning":

            model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = model_path, config=config)
        
        else:

            model = BertModel.from_pretrained(pretrained_model_name_or_path = model_path, config=config)

        self.model = model


class RoBERTa(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, mode: str = "masked_learning",
                 cls_token = True, optimizer = AdamW, loss = BCELoss,
                 **trainer_kwargs):
        
        self.config = RobertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            pad_token_id = 0
        )

        if mode == "masked_learning":

            model = RobertaForMaskedLM(self.config)
        
        else:

            model = RobertaModel(self.config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, mode = mode,
                 cls_token = cls_token, optimizer = optimizer, loss = loss, **trainer_kwargs)
        
    def _load(self, config, model_path, mode):
        self.config = config

        if mode == "masked_learning":

            model = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path = model_path, config=config)
        
        else:

            model = RobertaModel.from_pretrained(pretrained_model_name_or_path = model_path, config=config)

        self.model = model


class DeBERTa(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, mode: str = "masked_learning",
                 cls_token = True, optimizer = AdamW, loss = BCELoss,
                 **trainer_kwargs):
        
        self.config = DebertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            pad_token_id = 0
        )

        if mode == "masked_learning":

            model = DebertaForMaskedLM(self.config)
        
        else:

            model = DebertaModel(self.config)


        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, mode = mode,
                 cls_token = cls_token, optimizer = optimizer, loss = loss, **trainer_kwargs)
        
    def _load(self, config, model_path, mode):
        self.config = config

        if mode == "masked_learning":

            model = DebertaForMaskedLM.from_pretrained(pretrained_model_name_or_path = model_path, config=config)
        
        else:

            model = DebertaModel.from_pretrained(pretrained_model_name_or_path = model_path, config=config)

        self.model = model

    