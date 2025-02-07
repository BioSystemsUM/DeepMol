


import os
import shutil
from unittest import TestCase, skip

from deepmol.models.transformer_models import BERT, ModernBERT, RoBERTa, DeBERTa, TransformerModelForMaskedLM
from tests.integration_tests.dataset.test_dataset import TestDataset

# @skip
class TestMaskedLearningModels(TestDataset, TestCase):

    def test_fit_predict_evaluate(self):
        # ModernBERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        # BERT(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        # RoBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        model = DeBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)

        model.save("test")
        model = TransformerModelForMaskedLM.load("test")
        if os.path.exists('vocab.txt'):
            os.remove('vocab.txt')

        if os.path.exists('test'):
            os.removedirs('test')

    def test_fine_tuning(self):
        self.dataset_for_masked_learning.mask = False
        model = DeBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, accelerator="gpu", mode="classification",
                        devices=[0], max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)
        
        predictions = model.predict(self.dataset_for_masked_learning)

        self.assertEqual(predictions.shape[0], self.dataset_for_masked_learning.y.shape[0])
        self.assertEqual(predictions.shape[1], self.dataset_for_masked_learning.y.shape[1])

        if os.path.exists('vocab.txt'):
            os.remove('vocab.txt')

    def test_save_and_load(self):

        model = DeBERTa(vocab_size=self.dataset_for_masked_learning.tokenizer.vocab_size, 
                        accelerator="cpu", max_epochs=3, batch_size=56).fit(self.dataset_for_masked_learning)

        model.save("test")
        model = DeBERTa.load("test")
        model.mode = "classification"

        self.dataset_for_masked_learning.mask = False

        model.fit(self.dataset_for_masked_learning)

        predictions = model.predict(self.dataset_for_masked_learning)

        self.assertEqual(predictions.shape[0], self.dataset_for_masked_learning.y.shape[0])
        self.assertEqual(predictions.shape[1], self.dataset_for_masked_learning.y.shape[1])

        model.save("test")
        model = DeBERTa.load("test")
        
        if os.path.exists('vocab.txt'):
            os.remove('vocab.txt')

        if os.path.exists('test'):
            shutil.rmtree('test')

        predictions_2 = model.predict(self.dataset_for_masked_learning)

        self.assertEqual(predictions_2.shape[0], self.dataset_for_masked_learning.y.shape[0])
        self.assertEqual(predictions_2.shape[1], self.dataset_for_masked_learning.y.shape[1])

        import numpy as np
        np.testing.assert_array_equal(predictions, predictions_2)
        

        
