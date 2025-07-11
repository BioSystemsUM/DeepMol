# DeepMol Pipeline Optimization (AutoML)

In DeepMol we can optimize Pipelines using the `PipelineOptimization` class. This class
uses the `Optuna` library to optimize the hyperparameters of the pipeline. It is possible
to optimize between different `Transformers` (e.g. featurizers, scalers, etc.) and 
`Predictors` (i.e. models) and also to optimize their hyperparameters.

As with any optimization problem, pipeline optimization requires an objective function. By default, `DeepMol` sets the goal of maximizing or minimizing a chosen metric (such as accuracy) for a specified validation set. Nonetheless, users can integrate additional custom objective functions as needed.

Furthermore, a function can be created with the configuration space that will be optimized by `Optuna`. This function will
 receive a `Trial` object from `Optuna` and will return the
steps (i.e. `Transformers` and `Predictors`) of the pipeline. Alternatively, we can
use a set of predefined functions (presets) that are already implemented in
DeepMol. The following presets are available:
- `'sklearn'`: optimizes between all available Standardizers, Featurizers, Scalers,
            Feature Selectors in DeepMol and all available models in Sklearn.
- `'deepchem'`: optimizes between all available Standardizers and all available models
              in DeepChem (depending on the model different Featurizers, Scalers 
              and Feature Selectors may also be optimized).
- `'keras'`: optimizes between Standardizers, Featurizers, Scalers, Feature Selectors in 
           DeepMol and some base Keras models (1D CNNs, RNNs, Bidirectional RNNs and FCNNs).
- `'all'`: optimizes between all the above presets.

In the case of using a custom function, the function must return a list of tuples
where the first element of the tuple is the name of the step and the second element is the
object that implements the step. The pipeline class will respect the order of the steps you
return in the objective function. For example, if you return the following list of tuples:

```python
steps = [('featurizer', featurizer), ('scaler', scaler), ('model', model)]
```

The pipeline will first apply the featurizer, then the scaler and finally the model.


## Example using a custom steps function

In the following example we will assume that you already have the data processed (ready for
training). We will only use DeepMol's PipelineOptimization class to optimize the final
model. In this case we will optimize between a Random Forest and a SVC model. We will also
optimize the hyperparameters of the models.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.models import SklearnModel
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.splitters import RandomSplitter

from deepmol.pipeline_optimization.objective_wrapper import ObjectiveTrainEval

# DEFINE THE FUNCTION
def steps(trial):
    model = trial.suggest_categorical('model', ['RandomForestClassifier', 'SVC'])
    if model == 'RandomForestClassifier':
        n_estimators = trial.suggest_int('model__n_estimators', 10, 100, step=10)
        model = RandomForestClassifier(n_estimators=n_estimators)
    elif model == 'SVC':
        kernel = trial.suggest_categorical('model__kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        model = SVC(kernel=kernel)
    model = SklearnModel(model=model, model_dir='model')
    steps = [('model', model)]
    return steps
 
# LOAD THE DATA   
loader = CSVLoader('data_path...',
                   smiles_field='mols',
                   id_field='ids',
                   labels_fields=['y'],
                   features_fields=['f1', 'f2', 'f3', '...'],
                   mode='classification')
dataset_descriptors = loader.create_dataset(sep=",")
   
# OPTIMIZE THE PIPELINE 
po = PipelineOptimization(direction='maximize', study_name='test_predictor_pipeline')
metric = Metric(accuracy_score)
train, test = RandomSplitter().train_test_split(dataset_descriptors, seed=123)
po.optimize(train_dataset=train, test_dataset=test, objective_steps=steps, objective = ObjectiveTrainEval,
            metric=metric, n_trials=5, save_top_n=3, trial_timeout=600)
``` 

This will optimize between the objective function and will save the top 3 pipelines
(save_top_n=3) in the `study_name` folder. The `direction` parameter indicates if we want
to maximize or minimize the metric. In this case we want to maximize the accuracy. The
`n_trials` parameter indicates the number of trials that will be performed by `Optuna`. 
Moreover, if the `trial_timeout` parameter is set, the trial will stop after the
given number of seconds.

Additionally, we can also provide a storage (see: https://optuna.readthedocs.io/en/stable/reference/storages.html),
a sampler (https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) and a
pruner (https://optuna.readthedocs.io/en/stable/reference/pruners.html) to use in the
optimization.

The best parameters can be retrieved using the `best_params` property:

```python
best_params = po.best_params
```

The best trial can be retrieved using the `best_trial` property:

```python
best_trial = po.best_trial
```

The best score can be retrieved using the `best_value` property:

```python
best_score = po.best_value
```

All the trials can be retrieved using the `trials` property:

```python
trials = po.trials
```

The best pipeline can be retrieved using the `best_pipeline` property:

```python
best_pipeline = po.best_pipeline
```

More information about the trials and respective scores and parameters can be retrieved
using the `trials_dataframe` method.

```python
trials_df = po.trials_dataframe()
```

Information about the importance of the optimized parameters can be retrieved using the
`get_param_importances` method.

```python
param_importances = po.get_param_importances()
```

## Example using a preset

The following example assumes that you are providing raw SMILES data and respective labels.
We will use the `'all'` preset.

```python
from deepmol.loaders import CSVLoader
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.splitters import RandomSplitter
from sklearn.metrics import mean_squared_error
import optuna

from deepmol.pipeline_optimization.objective_wrapper import ObjectiveTrainEval

# LOAD THE DATA
loader = CSVLoader('dataset_regression_path',
                   smiles_field='smiles',
                   labels_fields=['pIC50'],
                   mode='regression')
dataset_regression = loader.create_dataset(sep=",")

# OPTIMIZE THE PIPELINE
po = PipelineOptimization(direction='minimize', study_name='test_pipeline', sampler=optuna.samplers.TPESampler(seed=42),
                          storage='sqlite:///my_experience.db')
metric = Metric(mean_squared_error)
train, test = RandomSplitter().train_test_split(dataset_regression, seed=123)
po.optimize(train_dataset=train, test_dataset=test, objective_steps='all', 
            metric=metric, n_trials=10, data=train, save_top_n=2, trial_timeout=600, 
            objective = ObjectiveTrainEval)
            
```

In this case we are optimizing between all the available steps in DeepMol.
In this case we want to minimize the mean squared error (regression task).

The best params and best pipeline can be obtained as in the previous example.


## Implement a custom objective class

In the following example we will implement a custom objective class. By default, the
objective class is `ObjectiveTrainEval`. This class will train the pipeline using the
train dataset and will evaluate the pipeline using the validation dataset. 
A new objective class must inherit from `Objective` and must implement the `__call__`
method. This method will receive a `Trial` object from `Optuna` and must return a
dictionary with the following keys:
- `'objective_steps'`: a list of tuples where the first element of the tuple is the name of the step
            and the second element is the object that implements the step.
- `'study'`: the `Study` object from `Optuna`.
- `'direction'`: the direction of the optimization (i.e. maximize or minimize).
- `'save_top_n'`: the number of top pipelines to save.
- `'trial_timeout'`: the number of seconds the trial has to run.

Here is an example of a custom objective function:

```python
from deepmol.pipeline_optimization.objective_wrapper import Objective

class MyObjective(Objective):
    """
    
    Parameters
    ----------
    objective_steps : callable
        Function that returns the steps of the pipeline for a given trial.
    study : optuna.study.Study
        Study object that stores the optimization history.
    direction : str or optuna.study.StudyDirection
        Direction of the optimization (minimize or maximize).
    save_top_n : int
        Number of best pipelines to save.
    **kwargs
        Additional keyword arguments passed to the objective_steps function.
    """

    def __init__(self, objective_steps, study, direction, save_top_n,
                 trial_timeout=86400, **kwargs):
        """
        Initialize the objective function.
        """
        super().__init__(objective_steps, study, direction, save_top_n, trial_timeout=trial_timeout)
        self.metric = kwargs.pop('metric')
        # the datasets or any additional data can be accessed through kwargs. Here is an example:

        self.train_dataset = kwargs.pop('train_dataset')
        self.test_dataset = kwargs.pop('test_dataset')

        # and passed as input of the PipelineOptimization().optimize(data=data). See below in the example of optimization calling.

        self.kwargs = kwargs

    def _run(self, trial: Trial):
        """
        Call the objective function.
        """
        # train the pipeline 
        # compute a score using self.metric for the test/validation set
        return score


po = PipelineOptimization(direction='minimize', study_name='test_pipeline', sampler=optuna.samplers.TPESampler(seed=42),
                          storage='sqlite:///my_experience.db')
metric = Metric(mean_squared_error)
train, test = RandomSplitter().train_test_split(dataset_regression, seed=123)
po.optimize(train_dataset=train, test_dataset=test, objective_steps='all', 
            metric=metric, n_trials=10, data=train, save_top_n=2, trial_timeout=600, 
            objective = MyObjective)
```

One example is given in [here](https://github.com/BioSystemsUM/DeepMol/blob/master/src/deepmol/pipeline_optimization/objective_wrapper.py).