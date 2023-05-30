import os
from unittest import TestCase

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

from deepmol.compound_featurization import SmilesOneHotEncoder
from deepmol.loaders import CSVLoader
from deepmol.models import KerasModel

from tests import TEST_DIR


class TestOneHotEncoder(TestCase):

    def setUp(self) -> None:
        smiles_dataset_path = os.path.join(TEST_DIR, 'data')
        dataset_smiles = os.path.join(smiles_dataset_path, "balanced_mini_dataset.csv")
        loader = CSVLoader(dataset_smiles,
                           smiles_field='Smiles',
                           labels_fields=['Class'])
        self.dataset_smiles = loader.create_dataset(sep=";")

        def classification_rnn_builder(input_shape):
            # Define the model
            model = Sequential()
            model.add(LSTM(64, input_shape=input_shape))
            model.add(Dense(1, activation='sigmoid'))
            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        self.model_builder = classification_rnn_builder

    def tearDown(self) -> None:
        # remove logs (files starting with 'deepmol.log')
        for f in os.listdir():
            if f.startswith('deepmol.log'):
                os.remove(f)

    def test_smiles_character_level_tokenizer(self):
        ohe = SmilesOneHotEncoder()
        tokenized = ohe.fit_transform(self.dataset_smiles)
        self.assertEqual(len(tokenized), len(self.dataset_smiles))

        rnn_model = KerasModel(model_builder=self.model_builder, input_shape=ohe.shape)

        rnn_model.fit(tokenized)

        predictions = rnn_model.predict(tokenized)
        self.assertEqual(len(predictions), len(self.dataset_smiles))
