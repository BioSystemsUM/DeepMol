# Pipeline

DeepMol provides a pipeline to perform almost all the steps above in a sequence without
having to worry about the details of each step. The pipeline can be used to perform
a prediction pipeline (last step is a data predictor) or a data transformation pipeline
(all steps are data transformers). Transformers must implement the _fit and _transform
methods and predictors must implement the _fit and _predict methods.

```python
from deepmol.loaders import CSVLoader
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from deepmol.models import KerasModel
from deepmol.standardizer import BasicStandardizer
from deepmol.compound_featurization import MorganFingerprint
from deepmol.scalers import StandardScaler
from deepmol.feature_selection import KbestFS
from deepmol.pipeline import Pipeline
from deepmol.metrics import Metric
from sklearn.metrics import accuracy_score

loader_train = CSVLoader('data_train_path.csv',
                         smiles_field='Smiles',
                         labels_fields=['Class'])
train_dataset = loader_train.create_dataset(sep=";")

loader_test = CSVLoader('data_test_path.csv',
                        smiles_field='Smiles',
                        labels_fields=['Class'])
test_dataset = loader_train.create_dataset(sep=";")
        
def basic_classification_model_builder(input_shape):
  model = Sequential()
  model.add(Dense(10, input_shape=input_shape, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

keras_model = KerasModel(model_builder=basic_classification_model_builder, epochs=2, input_shape=(1024,))
        
steps = [('standardizer', BasicStandardizer()),
         ('featurizer', MorganFingerprint(size=1024)),
         ('scaler', StandardScaler()),
         ('feature_selector', KbestFS(k=10)),
         ('model', keras_model)]

pipeline = Pipeline(steps=steps, path='test_pipeline/')

pipeline.fit_transform(train_dataset)

predictions = pipeline.predict(test_dataset)
pipeline.evaluate(test_dataset, [Metric(accuracy_score)])

# save pipeline
pipeline.save()

# load pipeline
pipeline = Pipeline.load('test_pipeline/')
``` 