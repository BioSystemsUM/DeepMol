import os
from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.pipeline import Pipeline
from tests import TEST_DIR


class TestPipeline(TestCase):

    def setUp(self) -> None:
        data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(data_path, "small_train_dataset.csv")
        features = [f'feat_{i}' for i in range(1, 2049)]
        loader = CSVLoader(dataset, smiles_field='mols', id_field='ids', labels_fields=['y'], features_fields=features)
        self.dataset = loader.create_dataset(sep=",")

    def tearDown(self) -> None:
        pass

    def test_predictor_pipeline(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf, model_dir='model.pkl')
        pipeline = Pipeline(steps=[('model', model)])

        pipeline.save()

        pipeline2 = Pipeline.load(pipeline.path)
        self.assertFalse(pipeline2.is_fitted())

        pipeline.fit_transform(self.dataset)
        predictions = pipeline.predict(self.dataset)
        self.assertEqual(len(predictions), len(self.dataset))
        e1, e2 = pipeline.evaluate(self.dataset, [Metric(accuracy_score)])
        self.assertTrue('accuracy_score' in e1.keys())
        self.assertEqual(e2, {})

        pipeline.save()

        pipeline3 = Pipeline.load(pipeline.path)
        self.assertTrue(pipeline3.is_fitted())
