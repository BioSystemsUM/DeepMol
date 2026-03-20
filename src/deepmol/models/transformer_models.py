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
    BertForMaskedLM, RobertaForMaskedLM, RobertaConfig, DebertaForMaskedLM, DebertaConfig, \
    BertModel, DebertaModel, ModernBertModel



from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.nn import BCELoss

from deepmol.utils.utils import normalize_labels_shape

import torch
from torchmetrics import Metric

class MaskedSymbolRecoveryAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Store counts as torch tensors for distributed training compatibility
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, labels, logits, mask_indices):
        """
        input_ids: Tensor of shape (batch_size, seq_len), input tokens with masking
        labels: Tensor of shape (batch_size, seq_len), true token labels
        logits: Tensor of shape (batch_size, seq_len, vocab_size), model outputs
        mask_indices: Boolean tensor of shape (batch_size, seq_len), where True indicates masked positions
        """
        # Get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Select only masked token positions
        masked_labels = labels[mask_indices]
        predicted_tokens = predicted_ids[mask_indices]
        
        # Count correct predictions
        correct_preds = (masked_labels == predicted_tokens).sum()
        total_preds = mask_indices.sum()

        # Update metric state
        self.correct += correct_preds
        self.total += total_preds

    def compute(self):
        """ Compute final accuracy score """
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)



class TransformerModelForMaskedLM(LightningModule, Model):

    def __init__(self, model, learning_rate=1e-4, batch_size=8, model_dir = None, 
                 cls_token = True, output_neurons = 1, optimizer = AdamW, loss = BCELoss, metric = None,
                 dataset_mode = None, patience = 2, 
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
        self._output_neurons = output_neurons
        self.patience = patience

        self.model_dir_is_temp = False

        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        else:
            model_dir = tempfile.mkdtemp()
            self.model_dir_is_temp = True

        self.masked_tokens_recovery_accuracy = MaskedSymbolRecoveryAccuracy()

        self.metric = metric

        self._model_dir = model_dir
        self.model = model
        self.model_class = model.__class__

        self.optimizer = optimizer

        self.dataset_mode = dataset_mode

        self.freeze_transformer = False
        self.embedding = False


    def _forward_fine_tuning(self, input_ids, attention_mask):

        if self.freeze_transformer:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state = outputs.last_hidden_state
        if self.cls_token:
            cls_output = last_hidden_state[:, 0, :]
        else:
            cls_output = last_hidden_state[:, 1:, :].mean(dim=1)

        head_output = self.head(cls_output)

        return head_output
    
    def forward(self, input_ids, attention_mask, labels=None):

        if self.embedding:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state
            return last_hidden_state[:, 1:, :].mean(dim=1)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, labels, masked_indexes = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        outputs = outputs.logits
        accuracy = self.masked_tokens_recovery_accuracy(labels, outputs, masked_indexes)
        self.log('train_masked_tokens_recovery_accuracy', accuracy, on_epoch=True, 
                prog_bar=True, logger=True, sync_dist=True)
            

        self.log('train_loss', loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        
        if self.metric != None:
            if isinstance(self.metric, list):
                for metric in self.metric:
                    score = metric(outputs, labels)
                    self.log(f'train_{metric.__class__.__name__}', score, on_epoch=True,
                            prog_bar=True, logger=True, sync_dist=True)
            else:
                score = self.metric(outputs, labels)
                self.log(f'train_{self.metric.__class__.__name__}', score, on_epoch=True,
                        prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels, masked_indexes = batch

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        val_loss = outputs.loss
        outputs = outputs.logits
        accuracy = self.masked_tokens_recovery_accuracy(labels, outputs, masked_indexes)
        self.log('validation_masked_tokens_recovery_accuracy', accuracy, on_epoch=True, 
                prog_bar=True, logger=True, sync_dist=True)

        self.log('val_loss', val_loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        
        if self.metric != None:
            if isinstance(self.metric, list):
                for metric in self.metric:
                    score = metric(outputs, labels)
                    self.log(f'val_{metric.__class__.__name__}', score, on_epoch=True,
                            prog_bar=True, logger=True, sync_dist=True)
            else:
                score = self.metric(outputs, labels)
                self.log(f'val_{self.metric.__class__.__name__}', score, on_epoch=True,
                        prog_bar=True, logger=True, sync_dist=True)
            
        return val_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask, labels, masked_indexes = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):

        input_ids, attention_mask, labels, masked_indexes = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        val_loss = outputs.loss
        outputs = outputs.logits
        accuracy = self.masked_tokens_recovery_accuracy(labels, outputs, masked_indexes)
        self.log('test_masked_tokens_recovery_accuracy', accuracy, on_epoch=True, 
                prog_bar=True, logger=True, sync_dist=True)

        self.log('test_loss', val_loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        
        if self.metric != None:
            if isinstance(self.metric, list):
                for metric in self.metric:
                    score = metric(outputs, labels)
                    self.log(f'test_{metric.__class__.__name__}', score, on_epoch=True,
                            prog_bar=True, logger=True, sync_dist=True)
            else:
                score = self.metric(outputs, labels)
                self.log(f'test_{self.metric.__class__.__name__}', score, on_epoch=True,
                        prog_bar=True, logger=True, sync_dist=True)
                    
    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        trainer = Trainer(**self.trainer_kwargs)
        self = self.eval()
        return trainer.test(self, dataloaders=dataloader)
    
    def configure_optimizers(self):
        if self.freeze_transformer:
            return self.optimizer(self.head.parameters(), lr=self.learning_rate)
        else:
            return self.optimizer(self.parameters(), lr=self.learning_rate)
    
    def _fit(self, dataset: Dataset, validation_dataset: Dataset = None):

        if self.freeze_transformer:
            self.model = self.model.eval()
        else:
            self.model = self.model.train()

        self.dataset_mode = dataset.mode
        self.hparams["dataset_mode"] = self.dataset_mode

        if isinstance(self.dataset_mode, list):
            self.output_neurons = len(self.dataset_mode)
        else:
            self.output_neurons = 1

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        callbacks = []

        if validation_dataset is not None:
            val_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
            monitor = "val_loss"
            early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience, verbose=True)
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
            verbose=True
        )

        callbacks.append(checkpoint_callback)
        
        self.trainer = Trainer(**self.trainer_kwargs,
                     callbacks=callbacks)  # Use gpus=1 if you have a GPU available
        self.trainer.fit(self, dataloader, val_dataloader)

    def get_embedding(self, dataset: Dataset):
        self.embedding = True
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        trainer = Trainer(**self.trainer_kwargs)
        self = self.eval()
        predictions = trainer.predict(self, dataloader)
        self.embedding = False

        predictions = torch.cat(predictions)
        # convert to numpy array
        predictions = predictions.detach().cpu().numpy()

        return predictions


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

        if return_invalid:
            predictions = _return_invalid(dataset, predictions)
        return predictions

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
        trainer = Trainer(**self.trainer_kwargs)
        self = self.eval()
        predictions = trainer.predict(self, dataloader)

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

        if self.mode != "masked_learning":
            if len(predictions.shape) > 1:
                if predictions.shape != (len(dataset.mols), dataset.n_tasks):
                    predictions = normalize_labels_shape(predictions, dataset.n_tasks)

        if return_invalid:
            predictions = _return_invalid(dataset, predictions)

        return predictions
    
    def save(self, file_path: str = None, vocab_path: str = None):

        if vocab_path is None:
            vocab_path = "vocab.txt"

        if file_path is None:
            pl_model_path = os.path.join(self.model_dir, "model.ckpt")
            model_path = os.path.join(self.model_dir, "model.pt")
            config_path = os.path.join(self.model_dir, "config.json")
            # copy vocab file to model dir
            with open(vocab_path, "rb") as f:
                vocab = f.read()
            with open(os.path.join(self.model_dir, "vocab.txt"), "wb") as f:
                f.write(vocab)
        else:
            os.makedirs(file_path, exist_ok=True)
            pl_model_path = os.path.join(file_path, "model.ckpt")
            model_path = os.path.join(file_path, "model.pt")
            config_path = os.path.join(file_path, "config.json")
            with open(vocab_path, "rb") as f:
                vocab = f.read()
            with open(os.path.join(file_path, "vocab.txt"), "wb") as f:
                f.write(vocab)
            print("Vocab file saved to: ", os.path.join(file_path, "vocab.txt"))
        # Save the model configuration
        self.config.to_json_file(config_path)

        self.trainer.save_checkpoint(pl_model_path)
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def load(cls, folder_path: str, **kwargs) -> 'TransformerModelForMaskedLM':
        pass

class ModernBERT(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8,
                 cls_token = True, optimizer = AdamW, loss = BCELoss, patience = 2,
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

        model = ModernBertForMaskedLM(self.config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate,
                 cls_token = cls_token, optimizer = optimizer, loss = loss, patience = patience, **trainer_kwargs)

    @classmethod
    def load(cls, folder_path: str, **kwargs) -> 'TransformerModelForMaskedLM':
        new_model = cls.load_from_checkpoint(os.path.join(folder_path, "model.ckpt"),  *kwargs)

        return new_model
        

class BERT(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, 
                 cls_token = True, optimizer = AdamW, loss = BCELoss, patience = 2, intermediate_size=None, max_position_embeddings=None,
                 **trainer_kwargs):
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        if max_position_embeddings is None:
            max_position_embeddings = max_length
        
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id = 0
        )

        model = BertForMaskedLM(self.config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate,
                 cls_token = cls_token, optimizer = optimizer, loss = loss, patience = patience, **trainer_kwargs)

    @classmethod
    def load(cls, folder_path: str, **kwargs) -> 'TransformerModelForMaskedLM':
        new_model = cls.load_from_checkpoint(os.path.join(folder_path, "model.ckpt"),  *kwargs)

        return new_model
        

class RoBERTa(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, patience = 2,
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


        model = RobertaForMaskedLM(self.config)
        

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, 
                 cls_token = cls_token, optimizer = optimizer, loss = loss, patience = patience, **trainer_kwargs)
        

class DeBERTa(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8,
                 cls_token = True, optimizer = AdamW, loss = BCELoss, patience = 2,
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


        model = DebertaForMaskedLM(self.config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, 
                 cls_token = cls_token, optimizer = optimizer, loss = loss, patience = patience, **trainer_kwargs)

    @classmethod
    def load(cls, folder_path: str, **kwargs) -> 'TransformerModelForMaskedLM':
        new_model = cls.load_from_checkpoint(os.path.join(folder_path, "model.ckpt"),  *kwargs)

        return new_model
        
    