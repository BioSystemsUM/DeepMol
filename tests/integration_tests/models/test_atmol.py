import os
import shutil
from unittest import TestCase, skip

from torchmetrics import Accuracy

from deepmol.models.atmol.atmol import AtMolLightning
from deepmol.models.atmol.utils_gat_pretrain import AtmolTorchDataset
from tests.integration_tests.dataset.test_dataset import TestDataset

class TestATMOL(TestDataset, TestCase):

    def test_featurize(self):
        AtmolTorchDataset(self.small_dataset_to_test).featurize()

    def test_fit(self):

        AtMolLightning(batch_size=2, max_epochs=4, accelerator="cpu").fit(self.small_dataset_to_test, validation_dataset=self.small_dataset_to_test)

    def test_export_data_and_import_to_model(self):
        
        dataset = AtmolTorchDataset(self.small_dataset_to_test).featurize()
        dataset.export("test.pt")

        dataset = AtmolTorchDataset.from_pt("test.pt")

        AtMolLightning(max_epochs=4, accelerator="cpu").fit(dataset, validation_dataset=dataset)

        if os.path.exists("test.pt"):
            os.remove("test.pt")

    def test_fit_save_and_load(self):
        from torchmetrics.classification import  MultilabelF1Score
        
        dataset = AtmolTorchDataset(self.multilabel_classification).featurize()
        dataset.export("test.pt")

        dataset = AtmolTorchDataset.from_pt("test.pt")
        self.assertEqual(type(dataset.mode), list)

        model = AtMolLightning(max_epochs=2, accelerator="cpu", n_output=1024).fit(dataset)

        model.save("test")
        model = AtMolLightning.load("test")
        model.mode = "classification"
        model.metric = MultilabelF1Score(average="macro", num_labels=23, threshold=0.5)
        
        model.fit(dataset)
        print(model.predict(dataset))

        if os.path.exists("test"):
            shutil.rmtree("test")




