from unittest import TestCase, skip

from sklearn.ensemble import RandomForestClassifier

from deepmol.loaders import CSVLoader
from deepmol.models import SklearnModel
from deepmol.pipeline import Pipeline
from deepmol.standardizer import BasicStandardizer


@skip
class TestPredict(TestCase):

    def test_create_pipeline(self):
        std = BasicStandardizer()
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf, model_dir='model')
        pipeline = Pipeline(steps=[('std', std), ('model', model)], path='test_predictor_pipeline/')
        pipeline.save()

    def test_load_pipeline(self):
        pipeline = Pipeline.load('test_predictor_pipeline')

        pipeline = Pipeline.load('heavy_test/test_predictor_pipeline')

    def test_predict(self):
        dataset = CSVLoader(smiles_field='smiles', id_field='ids', dataset_path="heavy_test/lotus_dataset_50000.csv"). \
            create_dataset()

        pipeline = Pipeline.load("heavy_test/trial_485")

        pipeline.predict(dataset)
