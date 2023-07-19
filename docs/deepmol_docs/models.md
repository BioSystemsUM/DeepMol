# Machine Learning models

## Scikit-learn models

<font size="4"> **Let's start by loading the data and splitting it into train and test sets** </font>


```python
from deepmol.splitters import RandomSplitter
from deepmol.loaders import SDFLoader

dataset = SDFLoader("../data/CHEMBL217_conformers.sdf", id_field="_ID", labels_fields=["_Class"]).create_dataset()
random_splitter = RandomSplitter()
train_dataset, test_dataset = random_splitter.train_test_split(dataset, frac_train=0.8)
```

    2023-06-01 13:38:42,546 — INFO — Assuming classification since there are less than 10 unique y values. If otherwise, explicitly set the mode to 'regression'!



```python
train_dataset.get_shape()
```

    ((13298,), None, (13298,))


<font size="4"> **Let's generate Morgan fingerprints from our data** </font>


```python
from deepmol.compound_featurization import MorganFingerprint

MorganFingerprint(n_jobs=10).featurize(train_dataset, inplace=True)
MorganFingerprint(n_jobs=10).featurize(test_dataset, inplace=True)
```

<font size="4"> **Now that we have our data ready, let's train a Random Forest model** </font>

### Train models
```python
from deepmol.models import SklearnModel
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
model = SklearnModel(model=rf)
model.fit(train_dataset)
```

### Predict
Now that we have our model trained, let's make some predictions**


```python
model.predict(test_dataset)
```




    array([[0.8 , 0.2 ],
           [1.  , 0.  ],
           [0.  , 1.  ],
           ...,
           [0.01, 0.99],
           [0.35, 0.65],
           [0.  , 1.  ]])


<font size="4"> **And finally, let's evaluate our model according to some metrics** </font>

### Evaluate

```python
from deepmol.metrics import Metric
from sklearn.metrics import roc_auc_score, accuracy_score

model.evaluate(test_dataset, metrics=[Metric(metric=roc_auc_score), Metric(metric=accuracy_score)])
```




    ({'roc_auc_score': 0.9977580691051172, 'accuracy_score': 0.9828571428571429},
     {})


### Save and load models
DeepMol also allows you to save your models without any effort.


```python
model.save("my_model")
```

Load them back is also very simple.

```python
from deepmol.models import SklearnModel
from deepmol.metrics import Metric
from sklearn.metrics import roc_auc_score, accuracy_score

model = SklearnModel.load("my_model")
model.evaluate(test_dataset, metrics=[Metric(metric=roc_auc_score), Metric(metric=accuracy_score)])
```




    ({'roc_auc_score': 0.9977580691051172, 'accuracy_score': 0.9828571428571429},
     {})



As you see in the previous example, DeepMol allows you to train and evaluate your models in a very simple way. You can also use any other model from Scikit-learn, such as SVMs, Logistic Regression, etc. You can also use any other featurization method from DeepMol, such as ECFP, GraphConv, etc. Moreover, saving and deploying your models never was so easy!

## Keras models

<font size="4"> **Let's start by extracting some features from our data** </font>


```python
from deepmol.compound_featurization import MorganFingerprint

MorganFingerprint(n_jobs=10).featurize(train_dataset, inplace=True)
MorganFingerprint(n_jobs=10).featurize(test_dataset, inplace=True)
```
### Train models

Now that we have our data ready, let's train a Deep Learning model
In DeepMol we provide full flexibility to the user to define the architecture of the model. The only requirement is that the model must be defined as a function that takes as input the input dimension of the data and returns a compiled Keras model. The function can also take as input any other parameter that the user wants to tune. In this case, we will define a simple model with two hidden layers and a dropout layer.


```python
from keras.layers import Dense, Dropout
from keras import Sequential

def create_model(input_dim, optimizer='adam', dropout=0.5):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

```

Now that we implemented our model, we can train it


```python
from deepmol.models import KerasModel

input_dim = train_dataset.X.shape[1]
model = KerasModel(create_model, epochs = 5, verbose=1, optimizer='adam', input_dim=input_dim)
model = model.fit(train_dataset)
```

    Epoch 1/5
    1330/1330 [==============================] - 2s 791us/step - loss: 0.2303 - accuracy: 0.9257
    Epoch 2/5
    1330/1330 [==============================] - 1s 777us/step - loss: 0.0879 - accuracy: 0.9741
    Epoch 3/5
    1330/1330 [==============================] - 1s 797us/step - loss: 0.0636 - accuracy: 0.9812
    Epoch 4/5
    1330/1330 [==============================] - 1s 798us/step - loss: 0.0544 - accuracy: 0.9847
    Epoch 5/5
    1330/1330 [==============================] - 1s 790us/step - loss: 0.0469 - accuracy: 0.9868


### Predict

```python
model.predict(test_dataset)
```

    104/104 [==============================] - 0s 565us/step

    array([[9.8671627e-01, 1.3283747e-02],
           [1.0000000e+00, 4.9822679e-10],
           [5.4292679e-03, 9.9457073e-01],
           ...,
           [5.3464174e-03, 9.9465358e-01],
           [1.9562900e-02, 9.8043710e-01],
           [6.4892769e-03, 9.9351072e-01]], dtype=float32)



### Evaluate

```python
model.evaluate(test_dataset, metrics=[Metric(metric=roc_auc_score), Metric(metric=accuracy_score)])
```

    104/104 [==============================] - 0s 585us/step

    ({'roc_auc_score': 0.9959412851927224, 'accuracy_score': 0.9795488721804512},
     {})



### Save and load models

```python
model.save("my_model")
```


```python
from deepmol.models import KerasModel
from deepmol.metrics import Metric
from sklearn.metrics import roc_auc_score, accuracy_score

model = KerasModel.load("my_model")
model.evaluate(test_dataset, metrics=[Metric(metric=roc_auc_score), Metric(metric=accuracy_score)])
```

    104/104 [==============================] - 0s 569us/step

    ({'roc_auc_score': 0.9959412851927224, 'accuracy_score': 0.9795488721804512},
     {})



## DeepChem models

<font size="4"> **Generate molecular graphs** </font>

```python
from deepmol.compound_featurization import ConvMolFeat

ConvMolFeat(n_jobs=10).featurize(train_dataset, inplace=True)
ConvMolFeat(n_jobs=10).featurize(test_dataset, inplace=True)
```

### Train models

```python
from deepchem.models import GraphConvModel
from deepmol.models import DeepChemModel

model = DeepChemModel(model=GraphConvModel(graph_conv_layers=[32, 32], dense_layer_size=128, n_tasks=1), epochs=5, verbose=1)
model.fit(train_dataset)
```

### Predict

```python
model.predict(test_dataset)
```




    array([[9.9915403e-01, 8.4594759e-04],
           [9.9851429e-01, 1.4855991e-03],
           [5.3278193e-02, 9.4672173e-01],
           ...,
           [4.0388817e-04, 9.9959618e-01],
           [9.7295139e-03, 9.9027050e-01],
           [2.3896188e-02, 9.7610384e-01]], dtype=float32)



### Evaluate
```python
from deepmol.metrics import Metric
from sklearn.metrics import roc_auc_score, accuracy_score

model.evaluate(test_dataset, metrics=[Metric(metric=roc_auc_score), Metric(metric=accuracy_score)])
```




    ({'roc_auc_score': 0.9941217864209249, 'accuracy_score': 0.9720300751879699},
     {})



### Save and load models

```python
model.save("my_model")
```


```python
from deepmol.models import DeepChemModel
from deepmol.metrics import Metric
from sklearn.metrics import roc_auc_score, accuracy_score

model = DeepChemModel.load("my_model")
model.evaluate(test_dataset, metrics=[Metric(metric=roc_auc_score), Metric(metric=accuracy_score)])
```




    ({'roc_auc_score': 0.9941217864209249, 'accuracy_score': 0.9720300751879699},
     {})




