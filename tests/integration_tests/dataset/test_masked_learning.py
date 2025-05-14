


import os
import shutil
from unittest import TestCase, skip

from deepmol.models.transformer_models import BERT, ModernBERT, RoBERTa, DeBERTa
from tests.integration_tests.dataset.test_dataset import TestDataset

# @skip
class TestMaskedLearningModels(TestDataset, TestCase):

    def test_fit_predict_evaluate(self):
        # ModernBERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        # BERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        # RoBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        model = DeBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="cpu", max_epochs=1, batch_size=56).fit(self.dataset_for_masked_learning)

        model.save("test")
        model = DeBERTa.load("test")
        if os.path.exists('vocab.txt'):
            os.remove('vocab.txt')

        if os.path.exists('test'):
            shutil.rmtree('test')


    def test_save_and_load(self):

        from torchmetrics.text import Perplexity

        model = BERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size,
                        accelerator="cpu" , max_epochs=1, batch_size=56, metric = Perplexity())

        model.fit(self.dataset_for_masked_learning)
        print(model.evaluate(self.dataset_for_masked_learning))
        model.save("test")

        if os.path.exists('test'):
            shutil.rmtree('test')


        

        
