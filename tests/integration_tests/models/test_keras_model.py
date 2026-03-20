from tests.unit_tests.models.test_models import ModelsTestCase
from unittest import TestCase, skip

from deepmol.datasets import SmilesDataset

import pandas as pd
import numpy as np

from tests import TEST_DIR

import os

#Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def top_k_categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

# Build model
# gpus = tf.config.list_physical_devices('GPU')
# if len(gpus) > 0:
#     tf.config.set_visible_devices(gpus[0], 'GPU')
# logical_gpus = tf.config.list_logical_devices('GPU')
# print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

# force cpu usage
tf.config.set_visible_devices([], 'GPU')

def model_build(): # num = number of categories
    input_f = layers.Input(shape=(2048,))
    # input_b = layers.Input(shape=(4096,))
    # input_fp = layers.Concatenate()([input_f,input_b])
    
    X = layers.Dense(2048, activation = 'relu')(input_f)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(3072, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.Dropout(0.2)(X)
    output = layers.Dense(730, activation = 'sigmoid')(X)
    model = keras.Model(inputs = input_f, outputs = output)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss=['binary_crossentropy'],metrics=['cosine_proximity',top_k_categorical_accuracy])
    return model



class TestKerasModel(TestCase):

    def setUp(self):
        dataset = pd.read_csv(os.path.join(TEST_DIR, 'data', 'np_dataset_small_sample.csv'))
        self.multi_task_dataset = SmilesDataset(smiles=dataset.SMILES.values, # only mandatory argument, a list of SMILES strings
                          mols=None,
                          ids=dataset.key.values,
                          X=None,
                          feature_names=None,
                          y=dataset.iloc[:,2:],
                          label_names=dataset.columns[2:],
                          mode='multilabel')
        
    def test_fit_model(self):
        from deepmol.compound_featurization import MorganFingerprint

        MorganFingerprint(n_jobs=-1).featurize(self.multi_task_dataset, inplace=True)

        from deepmol.models import KerasModel

        model = KerasModel(model_build, epochs = 5, verbose=1, mode=self.multi_task_dataset.mode)

        model.fit(self.multi_task_dataset)
        print(model.predict(self.multi_task_dataset).shape)