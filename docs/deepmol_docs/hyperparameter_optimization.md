# Hyperparameter optimization

<font size="4"> **First of all, let's load the data** </font>


```python
from deepmol.loaders import SDFLoader
from deepmol.splitters import RandomSplitter

dataset = SDFLoader("../data/CHEMBL217_conformers.sdf", id_field="_ID", labels_fields=["_Class"]).create_dataset()
train_dataset, valid_dataset, test_dataset = RandomSplitter().train_valid_test_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
```

    2023-06-02 11:34:09,557 — INFO — Assuming classification since there are less than 10 unique y values. If otherwise, explicitly set the mode to 'regression'!

<font size="4"> **Let's featurize the data** </font>

```python
from deepmol.compound_featurization import MorganFingerprint

morgan_fingerprints = MorganFingerprint()
morgan_fingerprints.featurize(train_dataset, inplace=True)
morgan_fingerprints.featurize(valid_dataset, inplace=True)
morgan_fingerprints.featurize(test_dataset, inplace=True)
```

## Hyperparameter tuning with scikit-learn

DeepMol provide methods to perform hyperparameter tuning with scikit-learn models. The hyperparameter tuning can be performed with a validation set, previously created, or with cross validation. Anyway, the hyperparameter tuning is performed with a random search if the number of iterations is specified, otherwise a grid search is performed.

Moreover, for each method the user have to specify the metric to optimize and if the metric has to be maximized or minimized. The user must specify the parameters to optimize and the values to try for each parameter.

The parameters must be specified as a dictionary, where the keys are the parameters names and the values are the values to try.


### Hyperparameter tuning with a validation set

Let's see how to perform hyperparameter tuning of a SVM with a validation set.

```python
from deepmol.parameter_optimization import HyperparameterOptimizerValidation
from sklearn.svm import SVC
from deepmol.metrics import Metric
from sklearn.metrics import accuracy_score

params_dict_svc = {"C": [1.0, 1.2, 0.8], "kernel": ['linear', 'poly', 'rbf']} # The keys are the parameters names and the values are the values to try
optimizer = HyperparameterOptimizerValidation(SVC,
                                              metric=Metric(accuracy_score),
                                              maximize_metric=True,
                                              n_iter_search=2,
                                              params_dict=params_dict_svc,
                                              model_type="sklearn")
best_svm, best_hyperparams, all_results = optimizer.fit(train_dataset=train_dataset, valid_dataset=valid_dataset)
```

In the end, we can check the performance of the best model on the test set.


```python
best_svm.evaluate(test_dataset, metrics = [Metric(accuracy_score)])
```




    ({'accuracy_score': 0.9879735417919423}, {})



We can also check the best combination of hyperparameters found.


```python
best_hyperparams
```




    {'C': 1.0, 'kernel': 'poly'}



We can also check the performance of all the models trained during the hyperparameter tuning. Each model is defined by the name of the parameters followed by the value of the parameter.


```python
all_results
```




    {'_C_1.000000_kernel_poly': 0.9885679903730445,
     '_C_0.800000_kernel_linear': 0.9765342960288809}



Finally, save your best model for deployment and new predictions!


```python
best_svm.save("my_model")
```

Bring it back to life and make predictions!

```python
from deepmol.models import SklearnModel

SklearnModel.load("../../examples/notebooks/my_model").predict(test_dataset)
```




    array([0., 0., 1., ..., 1., 0., 0.])



### Hyperparameter tuning with cross validation


```python
from deepmol.parameter_optimization import HyperparameterOptimizerCV
from sklearn.svm import SVC
from deepmol.metrics import Metric
from sklearn.metrics import accuracy_score

params_dict_svc = {"C": [1.0, 1.2, 0.8], "kernel": ['linear', 'poly', 'rbf']}
optimizer = HyperparameterOptimizerCV(SVC, metric=Metric(accuracy_score),
                                          maximize_metric=True,
                                          cv=3,
                                          n_iter_search=2,
                                          params_dict=params_dict_svc,
                                          model_type="sklearn")
best_svm, best_hyperparams, all_results = optimizer.fit(train_dataset=train_dataset)
```

    2023-06-01 16:19:58,650 — INFO — MODEL TYPE: sklearn
    2023-06-01 16:19:58,651 — INFO — Fitting 2 random models from a space of 9 possible models.
    2023-06-01 16:21:17,213 — INFO — 
     
     Best <function accuracy_score at 0x7f820037f0d0>: 0.973680 using {'kernel': 'linear', 'C': 1.0}
    2023-06-01 16:21:17,214 — INFO — 
     <function accuracy_score at 0x7f820037f0d0>: 0.973304 (0.000281) with: {'kernel': 'linear', 'C': 1.2} 
    
    2023-06-01 16:21:17,214 — INFO — 
     <function accuracy_score at 0x7f820037f0d0>: 0.973680 (0.000282) with: {'kernel': 'linear', 'C': 1.0} 
    
    2023-06-01 16:21:17,215 — INFO — Fitting best model!
    2023-06-01 16:21:32,759 — INFO — SklearnModel(mode='classification', model=SVC(kernel='linear'),
                 model_dir='/tmp/tmpom34y1bo')


Then, we can check the performance of the best model on the test set.


```python
best_svm.evaluate(test_dataset, metrics = [Metric(accuracy_score)])
```




    ({'accuracy_score': 0.9879735417919423}, {})



We can also check the best combination of hyperparameters found.


```python
best_hyperparams
```




    {'probability': True, 'kernel': 'poly', 'C': 1.2}



We can also check the performance of all the models trained during the hyperparameter tuning. Each model is defined by the name of the parameters followed by the value of the parameter.


```python
all_results
```




    {'mean_fit_time': array([103.76487271, 104.18803382]),
     'std_fit_time': array([1.19065455, 0.81880114]),
     'mean_score_time': array([8.71299458, 8.91998561]),
     'std_score_time': array([0.4336231 , 0.08003275]),
     'param_probability': masked_array(data=[True, True],
                  mask=[False, False],
            fill_value='?',
                 dtype=object),
     'param_kernel': masked_array(data=['poly', 'poly'],
                  mask=[False, False],
            fill_value='?',
                 dtype=object),
     'param_C': masked_array(data=[1.2, 1.0],
                  mask=[False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'probability': True, 'kernel': 'poly', 'C': 1.2},
      {'probability': True, 'kernel': 'poly', 'C': 1.0}],
     'split0_test_score': array([0.9835326 , 0.98217911]),
     'split1_test_score': array([0.98578841, 0.98511166]),
     'split2_test_score': array([0.98285199, 0.98217509]),
     'mean_test_score': array([0.98405766, 0.98315529]),
     'std_test_score': array([0.00125497, 0.00138337]),
     'rank_test_score': array([1, 2], dtype=int32)}



Finally, save your best model for deployment and new predictions!


```python
best_svm.save("my_model")
```

Bring it back to life and make predictions!


```python
from deepmol.models import SklearnModel

SklearnModel.load("my_model").predict(test_dataset)
```




    array([[9.99798926e-01, 2.01074317e-04],
           [9.99341488e-01, 6.58511735e-04],
           [3.02237487e-03, 9.96977625e-01],
           ...,
           [1.68278370e-08, 9.99999983e-01],
           [9.90501082e-01, 9.49891777e-03],
           [9.99827191e-01, 1.72809148e-04]])



## Hyperparameter tuning with keras

DeepMol provide methods to perform hyperparameter tuning with keras models. The hyperparameter tuning can be performed with a validation set, previously created, or with cross validation. Anyway, the hyperparameter tuning is performed with a random search if the number of iterations is specified, otherwise a grid search is performed.

As explained in the models section, to create a Keras model one have to define a function with the model architecture and the parameters to optimize.


```python
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_dim, optimizer='adam', dropout=0.5):
    # create model
    inputs = layers.Input(shape=input_dim)

    # Define the shared layers
    shared_layer_1 = layers.Dense(64, activation="relu")
    dropout_1 = Dropout(dropout)
    shared_layer_2 = layers.Dense(32, activation="relu")

    # Define the shared layers for the inputs
    x = shared_layer_1(inputs)
    x = dropout_1(x)
    x = shared_layer_2(x)

    task_output = layers.Dense(1, activation="sigmoid")(x)

    # Define the model that outputs the predictions for each task
    model = keras.Model(inputs=inputs, outputs=task_output)
    # Compile the model with different loss functions and metrics for each task
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model

```

### Hyperparameter tuning with a validation set

Let's see how to perform hyperparameter tuning of a DNN with a validation set.


```python
from deepmol.parameter_optimization import HyperparameterOptimizerValidation
from deepmol.metrics import Metric
from sklearn.metrics import accuracy_score

params_dict_dense = {
                   "input_dim": [train_dataset.X.shape[1]],
                   "dropout": [0.5, 0.6, 0.7],
                   "optimizer": ["adam"]
                   }

optimizer = HyperparameterOptimizerValidation(create_model,
                                              metric=Metric(accuracy_score),
                                              maximize_metric=True,
                                              n_iter_search=2,
                                              params_dict=params_dict_dense,
                                              model_type="keras")


best_dnn, best_hyperparams, all_results = optimizer.fit(train_dataset=train_dataset, valid_dataset=valid_dataset)
```

    2023-06-02 10:14:56,920 — INFO — Fitting 2 random models from a space of 3 possible models.
    2023-06-02 10:14:56,920 — INFO — Fitting model 1/2
    2023-06-02 10:14:56,921 — INFO — hyperparameters: {'input_dim': 2048, 'dropout': 0.5, 'optimizer': <class 'keras.optimizers.optimizer_experimental.adam.Adam'>}


    52/52 [==============================] - 0s 616us/step
    2023-06-02 10:18:08,223 — INFO — Model 1/2, Metric accuracy_score, Validation set 1: 0.981348
    2023-06-02 10:18:08,224 — INFO — 	best_validation_score so far: 0.981348
    2023-06-02 10:18:08,224 — INFO — Fitting model 2/2
    2023-06-02 10:18:08,224 — INFO — hyperparameters: {'input_dim': 2048, 'dropout': 0.6, 'optimizer': <class 'keras.optimizers.optimizer_experimental.adam.Adam'>}


    52/52 [==============================] - 0s 594us/step
    2023-06-02 10:21:16,230 — INFO — Model 2/2, Metric accuracy_score, Validation set 2: 0.983153
    2023-06-02 10:21:16,231 — INFO — 	best_validation_score so far: 0.983153


    416/416 [==============================] - 0s 616us/step
    2023-06-02 10:21:16,679 — INFO — Best hyperparameters: {'input_dim': 2048, 'dropout': 0.6, 'optimizer': <class 'keras.optimizers.optimizer_experimental.adam.Adam'>}
    2023-06-02 10:21:16,679 — INFO — train_score: 0.999850
    2023-06-02 10:21:16,679 — INFO — validation_score: 0.983153


In the end, we can check the performance of the best model on the test set.


```python
best_dnn.evaluate(test_dataset, metrics = [Metric(accuracy_score)])
```

    52/52 [==============================] - 0s 604us/step

    ({'accuracy_score': 0.9861695730607336}, {})



We can also check the best combination of hyperparameters found.


```python
best_hyperparams
```




    {'input_dim': 2048,
     'dropout': 0.6,
     'optimizer': keras.optimizers.optimizer_experimental.adam.Adam}



We can also check the performance of all the models trained during the hyperparameter tuning. Each model is defined by the name of the parameters followed by the value of the parameter.


```python
all_results
```




    {"_dropout_0.500000_input_dim_2048_optimizer_<class 'keras.optimizers.optimizer_experimental.adam.Adam'>": 0.9813477737665464,
     "_dropout_0.600000_input_dim_2048_optimizer_<class 'keras.optimizers.optimizer_experimental.adam.Adam'>": 0.9831528279181708}




```python
best_dnn
```


Finally, save your best model for deployment and new predictions!


```python
best_dnn.save("my_model")
```

Bring it back to life and make predictions!


```python
from deepmol.models import KerasModel

best_dnn = KerasModel.load("my_model")
```


```python
best_dnn.predict(test_dataset)
```

    52/52 [==============================] - 0s 609us/step

    array([[1.0000000e+00, 0.0000000e+00],
           [1.8446445e-03, 9.9815536e-01],
           [1.0000000e+00, 0.0000000e+00],
           ...,
           [3.2156706e-04, 9.9967843e-01],
           [1.0000000e+00, 0.0000000e+00],
           [1.0000000e+00, 0.0000000e+00]], dtype=float32)

### Hyperparameter tuning with cross validation

The hyperparameter tuning is very similar to the previous one, but in this case the hyperparameter tuning is performed with cross validation and as in for Sklearn models!

```python
from deepmol.parameter_optimization import HyperparameterOptimizerCV
from deepmol.metrics import Metric
from sklearn.metrics import accuracy_score

params_dict_dense = {
                   "input_dim": [train_dataset.X.shape[1]],
                   "dropout": [0.5, 0.6, 0.7],
                   "optimizer": ["adam"]
                   }

optimizer = HyperparameterOptimizerCV(create_model,
                                      metric=Metric(accuracy_score),
                                      maximize_metric=True,
                                      cv=3,
                                      n_iter_search=2,
                                      params_dict=params_dict_dense,
                                      model_type="keras")
best_dnn, best_hyperparams, all_results = optimizer.fit(train_dataset=train_dataset)
```

## Hyperparameter tuning with DeepChem models

DeepMol provide methods to perform hyperparameter tuning with deepchem models. The hyperparameter tuning can be performed with a validation set, previously created, or with cross validation. Anyway, the hyperparameter tuning is performed with a random search if the number of iterations is specified, otherwise a grid search is performed.

As explained in the models section, to create a DeepChem model one have to define a function with the model architecture and the parameters to optimize. It is MANDATORY to pass the **model_type** parameter to the model.

Let's see how to perform hyperparameter tuning of a GraphConvModel with a validation set.

First, we have to featurize the dataset. These features are specific for the model we want to train. In this case, we want to train a GraphConvModel, so we have to featurize the dataset with ConvMolFeat.
For more documentation on this matter check the [DeepChem documentation](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html).


```python
from deepmol.compound_featurization import ConvMolFeat

ConvMolFeat().featurize(train_dataset, inplace=True)
ConvMolFeat().featurize(valid_dataset, inplace=True)
ConvMolFeat().featurize(test_dataset, inplace=True)
```

### Hyperparameter tuning with validation set

```python
from deepmol.parameter_optimization import HyperparameterOptimizerValidation
from deepmol.models import DeepChemModel
from deepchem.models import GraphConvModel
from deepmol.metrics import Metric
from sklearn.metrics import accuracy_score

def graphconv_builder(graph_conv_layers, batch_size=256, epochs=5):
    graph = GraphConvModel
    return DeepChemModel(graph, epochs=epochs, n_tasks=1, graph_conv_layers=graph_conv_layers, batch_size=batch_size,
                           mode='classification')

model_graph = HyperparameterOptimizerValidation(model_builder=graphconv_builder,
                                                metric=Metric(accuracy_score),
                                                maximize_metric=True,
                                                n_iter_search=2,
                                                params_dict={'graph_conv_layers': [[64, 64], [32, 32]]},
                                                model_type="deepchem")

best_model, best_hyperparams, all_results = model_graph.fit(train_dataset=train_dataset,valid_dataset=valid_dataset)
```

In the end, we can check the performance of the best model on the test set.


```python
best_model.evaluate(test_dataset, metrics = [Metric(accuracy_score)])
```




    ({'accuracy_score': 0.9428743235117258}, {})



We can also check the best combination of hyperparameters found.


```python
best_hyperparams
```




    {'graph_conv_layers': [64, 64]}



We can also check the performance of all the models trained during the hyperparameter tuning. Each model is defined by the name of the parameters followed by the value of the parameter.


```python
all_results
```




    {'_graph_conv_layers_[64, 64]': 0.9494584837545126,
     '_graph_conv_layers_[32, 32]': 0.9199759326113117}



Finally, save your best model for deployment and new predictions!


```python
best_model.save("my_model")
```

Bring it back to life and make predictions!


```python
best_model = DeepChemModel.load("my_model")
```


```python
best_model.predict(test_dataset)
```




    array([[0.06773414, 0.9322658 ],
           [0.08101802, 0.9189819 ],
           [0.10755827, 0.8924417 ],
           ...,
           [0.00282426, 0.9971757 ],
           [0.00113406, 0.99886596],
           [0.98225605, 0.01774395]], dtype=float32)



### Hyperparameter tuning with cross validation


```python
from deepmol.parameter_optimization import HyperparameterOptimizerCV
from deepmol.models import DeepChemModel
from deepchem.models import GraphConvModel
from deepmol.metrics import Metric
from sklearn.metrics import roc_auc_score

def graphconv_builder(graph_conv_layers, batch_size=256, epochs=5):
    graph = GraphConvModel
    return DeepChemModel(graph, epochs=epochs, n_tasks=1, graph_conv_layers=graph_conv_layers, batch_size=batch_size,
                           mode='classification')

model_graph = HyperparameterOptimizerCV(model_builder=graphconv_builder,
                                        metric=Metric(roc_auc_score),
                                        n_iter_search=2,
                                        maximize_metric=True,
                                        cv = 2,
                                        params_dict={'graph_conv_layers': [[64, 64], [32, 32]]},
                                        model_type="deepchem")

best_model, best_hyperparameters, all_results = model_graph.fit(train_dataset=train_dataset)

```

In the end, we can check the performance of the best model on the test set.


```python
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from deepmol.metrics import Metric

test_preds = best_model.predict(test_dataset)

metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]

best_model.evaluate(test_dataset, metrics)

```




    ({'roc_auc_score': 0.9905489702285326,
      'precision_score': 0.9622166246851386,
      'accuracy_score': 0.9542994588093806},
     {})



We can also check the best combination of hyperparameters found.


```python
best_hyperparameters
```




    {'graph_conv_layers': [64, 64]}



We can also check the performance of all the models trained during the hyperparameter tuning. Each model is defined by the name of the parameters followed by the value of the parameter.


```python
all_results
```




    defaultdict(list,
                {'params': [{'graph_conv_layers': [64, 64]},
                  {'graph_conv_layers': [32, 32]}],
                 'mean_train_score': [0.9795196072718013, 0.980248484393072],
                 'mean_test_score': [0.9758974414293784, 0.9754465835745805],
                 'std_train_score': [0.003408778455179895, 0.000607063345097747],
                 'std_test_score': [0.0029443151372258725, 0.0011259723129122823],
                 'split0_train_score': [0.9829283857269812, 0.9796414210479742],
                 'split0_test_score': [0.9788417565666043, 0.9743206112616681],
                 'split1_train_score': [0.9761108288166214, 0.9808555477381697],
                 'split1_test_score': [0.9729531262921526, 0.9765725558874927]})




```python
all_results["params"]
```




    [{'graph_conv_layers': [64, 64]}, {'graph_conv_layers': [32, 32]}]



Finally, save your best model for deployment and new predictions!


```python
best_model.save("my_model")
```

Bring it back to life and make predictions!


```python
from deepmol.models import DeepChemModel

best_model = DeepChemModel.load("my_model")
```


```python
best_model.predict(test_dataset)
```




    array([[0.04331122, 0.95668876],
           [0.7563593 , 0.24364069],
           [0.00733165, 0.9926683 ],
           ...,
           [0.0438629 , 0.9561371 ],
           [0.06368961, 0.9363104 ],
           [0.9964748 , 0.00352522]], dtype=float32)


