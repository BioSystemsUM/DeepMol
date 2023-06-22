# Performing Feature Selection with DeepMol

The selection of the most relevant features can significantly improve the performance of a machine learning model in chemoinformatics. By removing irrelevant or redundant features, feature selection can reduce overfitting and improve the model's ability to generalize to new data. Additionally, feature selection can reduce the computational burden of training a machine learning model by reducing the number of features that need to be processed.

DeepMol supports many types of feature selection provided by scikit-learn including Low Variance Feature Selection, KBest, Percentile, Recursive Feature Elimination and selecting features based on importance weights.

<font size="5"> **Let's load our dataset with already computed features (2048 features)** </font>


```python
from deepmol.splitters import SingletaskStratifiedSplitter
from deepmol.loaders import CSVLoader

# Load data from CSV file
loader = CSVLoader(dataset_path='../data/example_data_with_features.csv',
                   smiles_field='mols',
                   id_field='ids',
                   labels_fields=['y'],
                   features_fields=[f'feat_{i+1}' for i in range(2048)],
                   shard_size=500,
                   mode='auto')
# create the dataset
csv_dataset = loader.create_dataset(sep=',', header=0)
csv_dataset.get_shape()
splitter = SingletaskStratifiedSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(csv_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
```

    2023-06-02 15:55:25,940 — INFO — Assuming classification since there are less than 10 unique y values. If otherwise, explicitly set the mode to 'regression'!
    2023-06-02 15:55:25,940 — INFO — Mols_shape: (500,)
    2023-06-02 15:55:25,941 — INFO — Features_shape: (500, 2048)
    2023-06-02 15:55:25,941 — INFO — Labels_shape: (500,)


## LowVarianceFS

Low variance feature selection is a technique used to select features in a dataset that have little or no variability across the data. This method is based on the assumption that features with low variance have little impact on the model's predictive ability and can be safely removed.

To apply low variance feature selection, one first calculates the variance of each feature across the entire dataset. The features with variance below a certain threshold are then removed from the dataset, typically by setting a minimum variance threshold or using a percentile of variance. The threshold value is usually determined by trial and error or through cross-validation.


```python
from copy import deepcopy
from deepmol.feature_selection import LowVarianceFS

# make a copy of our dataset
train_dataset_low_variance_fs = deepcopy(train_dataset)
test_dataset_low_variance_fs = deepcopy(test_dataset)

# instantiate our feature selector
fs = LowVarianceFS(threshold=0.15)
# perform feature selection
fs.fit_transform(train_dataset_low_variance_fs)
fs.transform(test_dataset_low_variance_fs)

train_dataset_low_variance_fs.get_shape()
test_dataset_low_variance_fs.get_shape()
```

    2023-06-02 15:55:26,049 — INFO — Mols_shape: (400,)
    2023-06-02 15:55:26,050 — INFO — Features_shape: (400, 47)
    2023-06-02 15:55:26,050 — INFO — Labels_shape: (400,)
    2023-06-02 15:55:26,051 — INFO — Mols_shape: (50,)
    2023-06-02 15:55:26,051 — INFO — Features_shape: (50, 47)
    2023-06-02 15:55:26,052 — INFO — Labels_shape: (50,)

    ((50,), (50, 47), (50,))



## KbestFS

SelectKBest is a feature selection algorithm in machine learning that selects the top k features with the highest predictive power from a given dataset. This algorithm works by scoring each feature and selecting the top k features based on their scores.

The score of each feature is determined using a statistical test, such as the chi-squared test, mutual information, or ANOVA F-test, depending on the nature of the dataset and the problem being solved. The algorithm computes a score for each feature, ranking them in descending order. It then selects the top k features with the highest scores and discards the rest.

The purpose of feature selection is to improve the model's performance by reducing the number of irrelevant or redundant features. Selecting only the most relevant features can help to reduce overfitting, increase model interpretability, and reduce computational costs.


```python
from sklearn.feature_selection import chi2
from deepmol.feature_selection import KbestFS

# make a copy of our dataset
train_dataset_kbest_fs = deepcopy(train_dataset)
test_dataset_kbest_fs = deepcopy(test_dataset)

fs = KbestFS(k=250, score_func=chi2) # the top k features with the highest predictive power will be kept
# perform feature selection

fs.fit_transform(train_dataset_kbest_fs)
fs.transform(test_dataset_kbest_fs)

train_dataset_kbest_fs.get_shape()
test_dataset_kbest_fs.get_shape()
```

    2023-06-02 15:55:26,154 — INFO — Mols_shape: (400,)
    2023-06-02 15:55:26,154 — INFO — Features_shape: (400, 250)
    2023-06-02 15:55:26,155 — INFO — Labels_shape: (400,)
    2023-06-02 15:55:26,155 — INFO — Mols_shape: (50,)
    2023-06-02 15:55:26,156 — INFO — Features_shape: (50, 250)
    2023-06-02 15:55:26,156 — INFO — Labels_shape: (50,)

    ((50,), (50, 250), (50,))



## PercentilFS

SelectPercentile is a feature selection algorithm in machine learning that selects the top features based on their statistical scores, similar to SelectKBest. However, instead of selecting a fixed number of features, SelectPercentile selects a percentage of the most informative features from a given dataset.

The main advantage of SelectPercentile over SelectKBest is that it adapts to datasets of different sizes, so it can select an appropriate number of features for datasets with different numbers of features.


```python
from deepmol.feature_selection import PercentilFS

# make a copy of our dataset
train_dataset_percentil_fs = deepcopy(train_dataset)
test_dataset_percentil_fs = deepcopy(test_dataset)

fs = PercentilFS(percentil=10, score_func=chi2) # keep the 10 percent top predictive features
fs.fit_transform(train_dataset_percentil_fs)
fs.transform(test_dataset_percentil_fs)

train_dataset_percentil_fs.get_shape()
test_dataset_percentil_fs.get_shape()
```

    2023-06-02 15:55:26,243 — INFO — Mols_shape: (400,)
    2023-06-02 15:55:26,244 — INFO — Features_shape: (400, 204)
    2023-06-02 15:55:26,245 — INFO — Labels_shape: (400,)
    2023-06-02 15:55:26,245 — INFO — Mols_shape: (50,)
    2023-06-02 15:55:26,245 — INFO — Features_shape: (50, 204)
    2023-06-02 15:55:26,246 — INFO — Labels_shape: (50,)

    ((50,), (50, 204), (50,))



## RFECVFS

Recursive Feature Elimination with Cross-Validation (RFECV) is a feature selection algorithm in machine learning that selects the most informative subset of features from a given dataset by iteratively eliminating the least important features.

RFECV uses a machine learning model (e.g., linear regression, logistic regression, or support vector machine) to rank the importance of each feature in the dataset. It then eliminates the feature with the lowest importance score, re-evaluates the performance of the model, and repeats the process until a specified number of features is reached.

The cross-validation (CV) component of RFECV involves dividing the dataset into k-folds, training the model on k-1 folds, and evaluating it on the remaining fold. This process is repeated k times, with each fold serving as the test set once. The performance of the model is then averaged over the k-folds, providing a more reliable estimate of model performance.


```python
from sklearn.ensemble import RandomForestClassifier
from deepmol.feature_selection import RFECVFS

train_dataset_RFECVFS = deepcopy(train_dataset)
test_dataset_RFECVFS = deepcopy(test_dataset)

fs = RFECVFS(estimator=RandomForestClassifier(n_jobs=-1), # model to use
             step=10, # number of features to remove at each step
             min_features_to_select=1024, # minimum number of feature to keep (it can have more than that but never less)
             cv=2, # number of folds in the cross validation
             verbose=3) # verbosity level

fs.fit_transform(train_dataset_RFECVFS)
fs.transform(test_dataset_RFECVFS)

train_dataset_RFECVFS.get_shape()
test_dataset_RFECVFS.get_shape()
```
    2023-06-02 15:56:08,628 — INFO — Mols_shape: (400,)
    2023-06-02 15:56:08,628 — INFO — Features_shape: (400, 1298)
    2023-06-02 15:56:08,629 — INFO — Labels_shape: (400,)
    2023-06-02 15:56:08,629 — INFO — Mols_shape: (50,)
    2023-06-02 15:56:08,629 — INFO — Features_shape: (50, 1298)
    2023-06-02 15:56:08,630 — INFO — Labels_shape: (50,)

    ((50,), (50, 1298), (50,))



## SelectFromModelFS

SelectFromModel is a feature selection algorithm in machine learning that selects the most informative subset of features from a given dataset based on the importance scores provided by a base estimator.

The algorithm works by training a machine learning model, such as a decision tree, random forest, or support vector machine, on the entire dataset and computing the importance score for each feature. The importance score reflects the contribution of each feature to the performance of the model.

SelectFromModel then selects the top features based on a threshold value specified by the user. The threshold value can be an absolute value or a percentile of the importance scores. Features with importance scores higher than the threshold value are retained, while those with scores lower than the threshold value are discarded.


```python
from deepmol.feature_selection import SelectFromModelFS

train_dataset_SelectFromModelFS = deepcopy(train_dataset)
test_dataset_SelectFromModelFS = deepcopy(test_dataset)

fs = SelectFromModelFS(estimator=RandomForestClassifier(n_jobs=-1), # model to use
                       threshold="mean") # Features whose importance is greater or equal are kept while the others are discarded. A percentil can also be used
                                         # In this case ("mean") will keep the features with importance higher than the mean and remove the others
fs.fit_transform(train_dataset_SelectFromModelFS)
fs.transform(test_dataset_SelectFromModelFS)

train_dataset_SelectFromModelFS.get_shape()
test_dataset_SelectFromModelFS.get_shape()
```

    2023-06-02 15:56:08,840 — INFO — Mols_shape: (400,)
    2023-06-02 15:56:08,840 — INFO — Features_shape: (400, 287)
    2023-06-02 15:56:08,841 — INFO — Labels_shape: (400,)
    2023-06-02 15:56:08,841 — INFO — Mols_shape: (50,)
    2023-06-02 15:56:08,841 — INFO — Features_shape: (50, 287)
    2023-06-02 15:56:08,842 — INFO — Labels_shape: (50,)

    ((50,), (50, 287), (50,))



## BorutaAlgorithm

The boruta algorithm works by comparing the importance of each feature in the original dataset with the importance of the same feature in a shuffled version of the dataset. If the importance of the feature in the original dataset is significantly higher than its importance in the shuffled dataset, the feature is deemed "confirmed" and is selected for the final feature subset.

It iteratively adds and removes features from the confirmed set until all features have been evaluated. The final set of confirmed features is the one that has a statistically significant higher importance in the original dataset compared to the shuffled dataset.

The advantage of Boruta is that it can capture complex relationships between features and identify interactions that may not be apparent in simpler feature selection methods. It can also handle missing values and noisy data, which can be challenging for other feature selection techniques.


```python
from deepmol.feature_selection import BorutaAlgorithm

train_dataset_boruta = deepcopy(train_dataset)
test_dataset_boruta = deepcopy(test_dataset)

fs = BorutaAlgorithm(estimator=RandomForestClassifier(n_jobs=-1), # model to use
                     task='classification') # classification or regression

fs.fit_transform(train_dataset_boruta)
fs.transform(test_dataset_boruta)

train_dataset_boruta.get_shape()
test_dataset_boruta.get_shape()
```

    2023-06-02 15:57:39,787 — INFO — Mols_shape: (400,)
    2023-06-02 15:57:39,787 — INFO — Features_shape: (400, 2048)
    2023-06-02 15:57:39,788 — INFO — Labels_shape: (400,)
    2023-06-02 15:57:39,788 — INFO — Mols_shape: (50,)
    2023-06-02 15:57:39,789 — INFO — Features_shape: (50, 2048)
    2023-06-02 15:57:39,789 — INFO — Labels_shape: (50,)

    ((50,), (50, 2048), (50,))


