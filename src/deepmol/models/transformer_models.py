import os
import pickle
import tempfile

from deepmol.loggers.logger import Logger
from deepmol.models.models import Model
from deepmol.datasets import Dataset

from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from torch.optim import AdamW

from transformers import ModernBertConfig, ModernBertForMaskedLM, BertConfig, BertForMaskedLM, RobertaForMaskedLM, RobertaConfig, DebertaForMaskedLM, DebertaConfig


from pytorch_lightning.callbacks import ModelCheckpoint


class TransformerModelForMaskedLM(LightningModule, Model):

    def __init__(self, model, learning_rate=1e-4, batch_size=8, model_dir = None,
                 **trainer_kwargs):
        
        # Initialize LightningModule
        LightningModule.__init__(self)
        self.model = model
        self.trainer_kwargs = trainer_kwargs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model_dir_is_temp = False

        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
        else:
            model_dir = tempfile.mkdtemp()
            self.model_dir_is_temp = True

        self._model_dir = model_dir
        self.model = model
        self.model_class = model.__class__
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log('train_loss', loss, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
    
    def _fit(self, dataset: Dataset):

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Create a checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath='checkpoints',
            filename='epoch-{epoch:02d}',
            save_top_k=5,
            every_n_epochs=1,
            verbose=True
        )

        self.trainer = Trainer(**self.trainer_kwargs,
                     callbacks=[checkpoint_callback])  # Use gpus=1 if you have a GPU available
        self.trainer.fit(self, dataloader)

    def _predict(self, dataset: Dataset):
        pass


    def save(self, file_path: str = None):

        if file_path is None:
            model_path = os.path.join(self.model_dir, "model.ckpt")
        else:
            os.makedirs(file_path, exist_ok=True)
            model_path = os.path.join(file_path, "model.ckpt")

        self.trainer.save_checkpoint(model_path)
        with open(os.path.join(file_path, "trainer.pk"), "w") as b:
            pickle.dump(self.trainer, b, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(file_path, "model.pk"), "w") as b:
            pickle.dump(self.model, b, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, folder_path: str) -> 'TransformerModelForMaskedLM':

        with open(os.path.join(folder_path, "model.pk"), "r") as b:
            model = pickle.load(b, protocol=pickle.HIGHEST_PROTOCOL)

        model_class = cls(model=model)

        with open(os.path.join(folder_path, "trainer.pk"), "r") as b:
        
            model_class.trainer = pickle.load(b, protocol=pickle.HIGHEST_PROTOCOL)

        model_path = os.path.join(folder_path, "model.ckpt")
        model_class.trainer.load_checkpoint(model_path)

        return model_class

class ModernBERT(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, 
                 **trainer_kwargs):
        
        config = ModernBertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            pad_token_id = 0
        )
        model = ModernBertForMaskedLM(config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, **trainer_kwargs)

class BERT(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, 
                 **trainer_kwargs):
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            pad_token_id = 0
        )
        model = BertForMaskedLM(config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, **trainer_kwargs)


class RoBERTa(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, 
                 **trainer_kwargs):
        
        config = RobertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            pad_token_id = 0
        )
        model = RobertaForMaskedLM(config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, **trainer_kwargs)


class DeBERTa(TransformerModelForMaskedLM):

    def __init__(self, vocab_size, max_length=256, hidden_size=256, num_hidden_layers=8, num_attention_heads=8,
                 learning_rate=1e-4, batch_size=8, 
                 **trainer_kwargs):
        
        config = DebertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            pad_token_id = 0
        )
        model = DebertaForMaskedLM(config)

        super().__init__(model, batch_size=batch_size, learning_rate=learning_rate, **trainer_kwargs)

    
