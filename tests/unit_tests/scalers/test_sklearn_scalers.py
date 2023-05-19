import copy
from unittest import TestCase, skip

from deepmol.scalers import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures, Normalizer, \
    Binarizer, KernelCenterer, QuantileTransformer, PowerTransformer
from unit_tests.scalers.test_scalers import ScalersTestCase


class SklearnScalersTestCase(ScalersTestCase, TestCase):

    @staticmethod
    def get_scalers():
        return [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), Normalizer(),
                Binarizer(), KernelCenterer(), QuantileTransformer(), PowerTransformer()]

    def test_scaler(self):
        for scaler in self.get_scalers():
            df = copy.deepcopy(self.dataset)
            df2 = copy.deepcopy(self.dataset)
            scaler.scale(df, inplace=True)
            # assert data has changed
            self.assertFalse((self.dataset.X == df.X).all())

            scaler.save("test_scaler.pkl")
            scaler.load("test_scaler.pkl")

            scaler.scale(df2, inplace=True)
            self.assertTrue((df.X == df2.X).all())

    @skip("Not implemented. This scaler changes the number of features!")
    def test_polynomial_features(self):
        df = copy.deepcopy(self.polynomial_features)
        scaler = PolynomialFeatures(degree=2)
        scaler.scale(df, inplace=True)
        self.assertFalse((self.polynomial_features.X == df.X).all())
