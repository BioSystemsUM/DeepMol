# User Input - Documentation

## Loading a Dataset
### Parameters
- **dataset**: dataset file (csv)
- **mols**: molecules column
- **id_field**: id column
- **labels_field**: labels column(s)
- **features_fields**: features column(s)
- **features2keep**: which features to keep 
- **shard_size**: chunk size to yield at one time

## Featurization
### Parameters
- **name**: which featurizer to use
- **type**: wether we want default or specify the parameters
- **params**: list of parameters (depends on featurizer)

### Supported featurizers
- All featurizers supported under Deep Mol's **rdkitFingerprints**

## Models
### Parameters
- **models**: list of models to compare
- **name**: model identifier
- **params**: depends on the model

#### Notes about model parameters
- '{}' means no params
- one value per parameter field means to run the model with those parameter values
- multiple values per parameter means to run parameter optimization and choose best combination

## Suported Models
- RandomForestClassifier
- C-Support Vector Classification
- KNeighborClassifier
- DecisionTreeClassifier
- RidgeClassifier
- SGDClassifier
- AdaBoostClassifier

## Metrics
- **metrics**: list of metrics to use (metrics included in DeepMol's Metrics)

# Example

```yaml
{
    "load": {
        "dataset": "preprocessed_dataset_wfoodb.csv",
        "mols": "Smiles",
        "labels_fields": "Class",
        "id_field": "ID"
    },
    "featurizer": {
        "name" : "morgan",
        "type" : "default",
        "params":{}
    },
    "models": [
        {"name": "RandomForestClassifier", "params": {"n_estimators": [5,25,50,100],
                                                      "criterion": ["entropy","gini"],
                                                      "max_features": ["auto", "sqrt", "log2", "None"]},
        {"name": "SVM", "params": {}}
    ],
    "metrics": ["roc_auc_score", "precision_score", "accuracy_score"]
}

``` 



