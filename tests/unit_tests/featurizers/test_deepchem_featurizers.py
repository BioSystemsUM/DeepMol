from copy import copy
from unittest import TestCase

from deepmol.compound_featurization import WeaveFeat, ConvMolFeat, MolGraphConvFeat, CoulombFeat, CoulombEigFeat, \
    SmileImageFeat, SmilesSeqFeat

from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestDeepChemFeaturizers(FeaturizerTestCase, TestCase):

    def validate_featurizer(self, featurizer, df, n_valid, **kwargs):
        featurizer(**kwargs).featurize(df)
        features = df.X
        self.assertEqual(len(features), n_valid)

    def test_featurize(self):
        df = copy(self.mock_dataset)
        valid = len(self.original_smiles)
        self.validate_featurizer(ConvMolFeat, df, valid)
        self.validate_featurizer(WeaveFeat, df, valid)
        self.validate_featurizer(MolGraphConvFeat, df, valid)
        self.validate_featurizer(CoulombFeat, df, valid, max_atoms=100)
        self.validate_featurizer(CoulombEigFeat, df, valid, max_atoms=100)
        self.validate_featurizer(SmileImageFeat, df, valid)
        self.validate_featurizer(SmilesSeqFeat, df, valid)

    def test_featurize_with_nan(self):
        df = copy(self.mock_dataset_with_invalid)
        valid = len(self.original_smiles_with_invalid) - 1
        self.validate_featurizer(ConvMolFeat, df, valid, n_jobs=1)
        self.validate_featurizer(WeaveFeat, df, valid)
        self.validate_featurizer(MolGraphConvFeat, df, valid, n_jobs=-1)
        self.validate_featurizer(CoulombFeat, df, valid, max_atoms=100, seed=123, n_jobs=3)
        self.validate_featurizer(CoulombEigFeat, df, valid, max_atoms=100, seed=123, n_jobs=2)
        self.validate_featurizer(SmileImageFeat, df, valid, n_jobs=1)
        self.validate_featurizer(SmilesSeqFeat, df, valid+1)
