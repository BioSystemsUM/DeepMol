# Performing Feature Selection with DeepMol

The selection of the most relevant features can significantly improve the performance of a machine learning model in chemoinformatics. By removing irrelevant or redundant features, feature selection can reduce overfitting and improve the model's ability to generalize to new data. Additionally, feature selection can reduce the computational burden of training a machine learning model by reducing the number of features that need to be processed.

DeepMol supports many types of feature selection provided by scikit-learn including Low Variance Feature Selection, KBest, Percentile, Recursive Feature Elimination and selecting features based on importance weights.

### Let's load our dataset with already computed features (2048 features)


```python
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
```

    2023-05-29 15:54:52,861 — INFO — Assuming classification since there are less than 10 unique y values. If otherwise, explicitly set the mode to 'regression'!
    2023-05-29 15:54:52,862 — INFO — Mols_shape: (500,)
    2023-05-29 15:54:52,862 — INFO — Features_shape: (500, 2048)
    2023-05-29 15:54:52,863 — INFO — Labels_shape: (500,)





    ((500,), (500, 2048), (500,))



### Lets use the LowVarianceFS feature selector

Low variance feature selection is a technique used to select features in a dataset that have little or no variability across the data. This method is based on the assumption that features with low variance have little impact on the model's predictive ability and can be safely removed.

To apply low variance feature selection, one first calculates the variance of each feature across the entire dataset. The features with variance below a certain threshold are then removed from the dataset, typically by setting a minimum variance threshold or using a percentile of variance. The threshold value is usually determined by trial and error or through cross-validation.


```python
from copy import deepcopy
from deepmol.feature_selection import LowVarianceFS

# make a copy of our dataset
d1 = deepcopy(csv_dataset)

# instantiate our feature selector
fs = LowVarianceFS(threshold=0.15)
# perform feature selection
fs.select_features(d1, inplace=True)
# see changes in the shape of the features
d1.get_shape() # our dataset only has 47 features (out of 2048) with a variability higher than 15%.
```

    2023-05-29 15:55:04,339 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:04,339 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:04,339 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:04,339 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:04,341 — INFO — Features_shape: (500, 49)
    2023-05-29 15:55:04,341 — INFO — Features_shape: (500, 49)
    2023-05-29 15:55:04,341 — INFO — Features_shape: (500, 49)
    2023-05-29 15:55:04,341 — INFO — Features_shape: (500, 49)
    2023-05-29 15:55:04,342 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:04,342 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:04,342 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:04,342 — INFO — Labels_shape: (500,)





    ((500,), (500, 49), (500,))



### Let's use the KbestFS feature selector

SelectKBest is a feature selection algorithm in machine learning that selects the top k features with the highest predictive power from a given dataset. This algorithm works by scoring each feature and selecting the top k features based on their scores.

The score of each feature is determined using a statistical test, such as the chi-squared test, mutual information, or ANOVA F-test, depending on the nature of the dataset and the problem being solved. The algorithm computes a score for each feature, ranking them in descending order. It then selects the top k features with the highest scores and discards the rest.

The purpose of feature selection is to improve the model's performance by reducing the number of irrelevant or redundant features. Selecting only the most relevant features can help to reduce overfitting, increase model interpretability, and reduce computational costs.


```python
from sklearn.feature_selection import chi2
from deepmol.feature_selection import KbestFS

# make a copy of our dataset
d2 = deepcopy(csv_dataset)
fs = KbestFS(k=250, score_func=chi2) # the top k features with the highest predictive power will be kept
# perform feature selection
fs.select_features(d2, inplace=True)
d2.get_shape() # as we can see only 250 feature were kept
```

    2023-05-29 15:55:15,409 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:15,409 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:15,409 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:15,409 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:15,409 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:15,414 — INFO — Features_shape: (500, 250)
    2023-05-29 15:55:15,414 — INFO — Features_shape: (500, 250)
    2023-05-29 15:55:15,414 — INFO — Features_shape: (500, 250)
    2023-05-29 15:55:15,414 — INFO — Features_shape: (500, 250)
    2023-05-29 15:55:15,414 — INFO — Features_shape: (500, 250)
    2023-05-29 15:55:15,416 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:15,416 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:15,416 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:15,416 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:15,416 — INFO — Labels_shape: (500,)





    ((500,), (500, 250), (500,))



### Let's use the PercentilFS feature selector

SelectPercentile is a feature selection algorithm in machine learning that selects the top features based on their statistical scores, similar to SelectKBest. However, instead of selecting a fixed number of features, SelectPercentile selects a percentage of the most informative features from a given dataset.

The main advantage of SelectPercentile over SelectKBest is that it adapts to datasets of different sizes, so it can select an appropriate number of features for datasets with different numbers of features.


```python
from deepmol.feature_selection import PercentilFS

# make a copy of our dataset
d3 = deepcopy(csv_dataset)
fs = PercentilFS(percentil=10, score_func=chi2) # keep the 10 percent top predictive features
fs.select_features(d3, inplace=True)
d3.get_shape() # 10 percent of 2048 features --> 204 features were kept
```

    2023-05-29 15:55:26,547 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:26,547 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:26,547 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:26,547 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:26,547 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:26,547 — INFO — Mols_shape: (500,)
    2023-05-29 15:55:26,549 — INFO — Features_shape: (500, 205)
    2023-05-29 15:55:26,549 — INFO — Features_shape: (500, 205)
    2023-05-29 15:55:26,549 — INFO — Features_shape: (500, 205)
    2023-05-29 15:55:26,549 — INFO — Features_shape: (500, 205)
    2023-05-29 15:55:26,549 — INFO — Features_shape: (500, 205)
    2023-05-29 15:55:26,549 — INFO — Features_shape: (500, 205)
    2023-05-29 15:55:26,551 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:26,551 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:26,551 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:26,551 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:26,551 — INFO — Labels_shape: (500,)
    2023-05-29 15:55:26,551 — INFO — Labels_shape: (500,)





    ((500,), (500, 205), (500,))



### Let's use the RFECVFS feature selector

Recursive Feature Elimination with Cross-Validation (RFECV) is a feature selection algorithm in machine learning that selects the most informative subset of features from a given dataset by iteratively eliminating the least important features.

RFECV uses a machine learning model (e.g., linear regression, logistic regression, or support vector machine) to rank the importance of each feature in the dataset. It then eliminates the feature with the lowest importance score, re-evaluates the performance of the model, and repeats the process until a specified number of features is reached.

The cross-validation (CV) component of RFECV involves dividing the dataset into k-folds, training the model on k-1 folds, and evaluating it on the remaining fold. This process is repeated k times, with each fold serving as the test set once. The performance of the model is then averaged over the k-folds, providing a more reliable estimate of model performance.


```python
from sklearn.ensemble import RandomForestClassifier
from deepmol.feature_selection import RFECVFS

d4 = deepcopy(csv_dataset)
fs = RFECVFS(estimator=RandomForestClassifier(n_jobs=-1), # model to use
             step=10, # number of features to remove at each step
             min_features_to_select=1024, # minimum number of feature to keep (it can have more than that but never less)
             cv=2, # number of folds in the cross validation
             verbose=3) # verbosity level
fs.select_features(d4, inplace=True)
```


```python
d4.get_shape()
```

    2023-05-29 15:56:20,221 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:20,221 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:20,221 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:20,221 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:20,221 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:20,221 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:20,221 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:20,224 — INFO — Features_shape: (500, 1158)
    2023-05-29 15:56:20,224 — INFO — Features_shape: (500, 1158)
    2023-05-29 15:56:20,224 — INFO — Features_shape: (500, 1158)
    2023-05-29 15:56:20,224 — INFO — Features_shape: (500, 1158)
    2023-05-29 15:56:20,224 — INFO — Features_shape: (500, 1158)
    2023-05-29 15:56:20,224 — INFO — Features_shape: (500, 1158)
    2023-05-29 15:56:20,224 — INFO — Features_shape: (500, 1158)
    2023-05-29 15:56:20,226 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:20,226 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:20,226 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:20,226 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:20,226 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:20,226 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:20,226 — INFO — Labels_shape: (500,)





    ((500,), (500, 1158), (500,))



### Let's use the SelectFromModelFS feature selector

SelectFromModel is a feature selection algorithm in machine learning that selects the most informative subset of features from a given dataset based on the importance scores provided by a base estimator.

The algorithm works by training a machine learning model, such as a decision tree, random forest, or support vector machine, on the entire dataset and computing the importance score for each feature. The importance score reflects the contribution of each feature to the performance of the model.

SelectFromModel then selects the top features based on a threshold value specified by the user. The threshold value can be an absolute value or a percentile of the importance scores. Features with importance scores higher than the threshold value are retained, while those with scores lower than the threshold value are discarded.


```python
from deepmol.feature_selection import SelectFromModelFS

d5 = deepcopy(csv_dataset)
fs = SelectFromModelFS(estimator=RandomForestClassifier(n_jobs=-1), # model to use
                       threshold="mean") # Features whose importance is greater or equal are kept while the others are discarded. A percentil can also be used
                                         # In this case ("mean") will keep the features with importance higher than the mean and remove the others
fs.select_features(d5, inplace=True)
d5.get_shape()
```

    2023-05-29 15:56:37,749 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:37,749 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:37,749 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:37,749 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:37,749 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:37,749 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:37,749 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:37,749 — INFO — Mols_shape: (500,)
    2023-05-29 15:56:37,752 — INFO — Features_shape: (500, 286)
    2023-05-29 15:56:37,752 — INFO — Features_shape: (500, 286)
    2023-05-29 15:56:37,752 — INFO — Features_shape: (500, 286)
    2023-05-29 15:56:37,752 — INFO — Features_shape: (500, 286)
    2023-05-29 15:56:37,752 — INFO — Features_shape: (500, 286)
    2023-05-29 15:56:37,752 — INFO — Features_shape: (500, 286)
    2023-05-29 15:56:37,752 — INFO — Features_shape: (500, 286)
    2023-05-29 15:56:37,752 — INFO — Features_shape: (500, 286)
    2023-05-29 15:56:37,755 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:37,755 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:37,755 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:37,755 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:37,755 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:37,755 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:37,755 — INFO — Labels_shape: (500,)
    2023-05-29 15:56:37,755 — INFO — Labels_shape: (500,)





    ((500,), (500, 286), (500,))



### Let's use the BorutaAlgorithm feature selector

The boruta algorithm works by comparing the importance of each feature in the original dataset with the importance of the same feature in a shuffled version of the dataset. If the importance of the feature in the original dataset is significantly higher than its importance in the shuffled dataset, the feature is deemed "confirmed" and is selected for the final feature subset.

It iteratively adds and removes features from the confirmed set until all features have been evaluated. The final set of confirmed features is the one that has a statistically significant higher importance in the original dataset compared to the shuffled dataset.

The advantage of Boruta is that it can capture complex relationships between features and identify interactions that may not be apparent in simpler feature selection methods. It can also handle missing values and noisy data, which can be challenging for other feature selection techniques.


```python
from deepmol.feature_selection import BorutaAlgorithm

d6 = deepcopy(csv_dataset)
fs = BorutaAlgorithm(estimator=RandomForestClassifier(n_jobs=-1), # model to use
                     task='classification') # classification or regression
fs.select_features(d6, inplace=True)
d6.get_shape()
```

    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,196 — INFO — Mols_shape: (500,)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,199 — INFO — Features_shape: (500, 74)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)
    2023-05-29 15:58:13,202 — INFO — Labels_shape: (500,)





    ((500,), (500, 74), (500,))


