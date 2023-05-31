# Imbalanced Learn with DeepMol

Imbalanced data is a common problem in machine learning and deep learning, including in the field of chemoinformatics. Imbalanced data refers to datasets where the number of instances of one class is significantly larger or smaller than the number of instances of other classes. For example, in chemoinformatics, there may be datasets where the number of active compounds is much smaller than the number of inactive compounds.

Imbalanced data can lead to biased and suboptimal models, as traditional machine learning algorithms may not be able to learn the minority class effectively. This is where imbalanced learning techniques, such as those implemented in the imbalanced-learn library, can be important.

DeepMol provides various methods for handling imbalanced data, including oversampling, undersampling, and combination methods. These techniques can help to balance the data and improve the performance of the machine learning models, especially in chemoinformatics where imbalanced data is common.

For example, in a chemoinformatics dataset where there are many inactive compounds and few active compounds, oversampling techniques such as SMOTE (Synthetic Minority Over-sampling Technique) can be used to generate synthetic examples of the minority class, improving the model's ability to learn from the minority class. On the other hand, undersampling techniques can be used to reduce the number of majority class samples, making the dataset more balanced.

### Let's create a dataset with imbalanced labels


```python
from deepmol.compound_featurization import TwoDimensionDescriptors
from collections import Counter
from deepmol.datasets import SmilesDataset

import pandas as pd

df = pd.read_csv('../data/CHEMBL217_reduced.csv', header=0)
# pick 100 cases where 'Activity_Flag' (label) is 1 and 1000 cases where 'Activity_Flag' is 0
# select 100 cases where the label is 1
cases_1 = df[df['Activity_Flag'] == 1].head(100)
# select 1000 cases where the label is 0
cases_0 = df[df['Activity_Flag'] == 0].head(1000)

unbalanced_data = pd.concat([cases_1, cases_0])

data = SmilesDataset(smiles=unbalanced_data.SMILES,
                     ids=unbalanced_data.Original_Entry_ID,
                     y=unbalanced_data.Activity_Flag,
                     label_names=['Activity_Flag'])
TwoDimensionDescriptors().featurize(data, inplace=True)


# count y values in dataset.y
Counter(data.y)
```

    Counter({1: 100, 0: 1000})



### Over Sampling Methods

### RandomOverSampler

The RandomOverSampler is a technique used to address the problem of imbalanced data in machine learning. It is a data augmentation technique that creates synthetic samples of the minority class by randomly duplicating existing samples until the number of samples in the minority class matches the number of samples in the majority class.

#### Sampling strategies
Sampling information to resample the data set.
When float, it corresponds to the desired ratio of the number of samples in the minority class over
the number of samples in the majority class after resampling.
When str, specify the class targeted by the resampling. The number of samples in the different classes
will be equalized. Possible choices are:
   - 'minority': resample only the minority class;
   - 'not minority': resample all classes but the minority class;
   - 'not majority': resample all classes but the majority class;
   - 'all': resample all classes;
   - 'auto': equivalent to 'not majority'.

When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
samples for each targeted class.
When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
The values correspond to the desired number of samples for each class.


```python
from copy import deepcopy
from deepmol.imbalanced_learn import RandomOverSampler

d1 = deepcopy(data)
sampler = RandomOverSampler(sampling_strategy=0.75, random_state=123)
d1 = sampler.sample(d1)

Counter(d1.y)
```




    Counter({1: 750, 0: 1000})



### SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) is another data augmentation technique used to address the problem of imbalanced data in machine learning. SMOTE is similar to the RandomOverSampler, but instead of randomly duplicating minority class samples, it creates synthetic samples by interpolating between pairs of minority class samples.

#### Sampling strategies
Sampling information to resample the data set.
When float, it corresponds to the desired ratio of the number of samples in the minority class over
the number of samples in the majority class after resampling.
When str, specify the class targeted by the resampling. The number of samples in the different classes
will be equalized. Possible choices are:
   - 'minority': resample only the minority class;
   - 'not minority': resample all classes but the minority class;
   - 'not majority': resample all classes but the majority class;
   - 'all': resample all classes;
   - 'auto': equivalent to 'not majority'.

When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
samples for each targeted class.
When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
The values correspond to the desired number of samples for each class.



```python
from deepmol.imbalanced_learn import SMOTE

d2 = deepcopy(data)
sampler = SMOTE(sampling_strategy=0.8, random_state=123, k_neighbors=5, n_jobs=-1)
d2 = sampler.sample(d2)

Counter(d2.y)
```

    Counter({1: 800, 0: 1000})



### Under Sampling Methods

### RandomUnderSampler

The RandomUnderSampler is a technique used to address the problem of imbalanced data in machine learning. It is a data reduction technique that reduces the number of samples in the majority class by randomly removing samples until the number of samples in the majority class matches the number of samples in the minority class.

#### Sampling strategies

Sampling information to resample the data set.
When float, it corresponds to the desired ratio of the number of samples in the minority class over
the number of samples in the majority class after resampling.
When str, specify the class targeted by the resampling. The number of samples in the different classes
will be equalized. Possible choices are:

   - 'minority': resample only the minority class;
   - 'not minority': resample all classes but the minority class;
   - 'not majority': resample all classes but the majority class;
   - 'all': resample all classes;
   - 'auto': equivalent to 'not majority'.

When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
samples for each targeted class.
When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
The values correspond to the desired number of samples for each class.



```python
from deepmol.imbalanced_learn import RandomUnderSampler

d3 = deepcopy(data)
sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=123, replacement=True)
d3 = sampler.sample(d3)

Counter(d3.y)
```




    Counter({0: 200, 1: 100})



### ClusterCentroids

ClusterCentroids is a technique used to address the problem of imbalanced data in machine learning. It is a data undersampling technique that creates synthetic samples of the majority class by clustering the majority class data and then generating centroids for each cluster. These centroids are then used as representative samples for the majority class.

#### Sampling strategies
Sampling information to resample the data set.
When float, it corresponds to the desired ratio of the number of samples in the minority class over
the number of samples in the majority class after resampling.
When str, specify the class targeted by the resampling. The number of samples in the different classes
will be equalized. Possible choices are:
   - 'minority': resample only the minority class;
   - 'not minority': resample all classes but the minority class;
   - 'not majority': resample all classes but the majority class;
   - 'all': resample all classes;
   - 'auto': equivalent to 'not majority'.

When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
samples for each targeted class.
When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
The values correspond to the desired number of samples for each class.


```python
from sklearn.cluster import KMeans
from deepmol.imbalanced_learn import ClusterCentroids

d4 = deepcopy(data)
sampler = ClusterCentroids(sampling_strategy=1, random_state=123, estimator=KMeans(), voting='hard')
d4 = sampler.sample(d4)

Counter(d4.y)
```




    Counter({0: 100, 1: 100})



### Combination of Under and Over Sampling

### SMOTEENN

SMOTEENN is a hybrid technique that combines two other techniques, SMOTE (Synthetic Minority Over-sampling Technique) and Edited Nearest Neighbors (ENN), to address the problem of imbalanced data in machine learning. SMOTE is used to oversample the minority class by creating synthetic samples, while ENN is used to undersample the majority class by removing samples that are misclassified by a k-NN classifier.

#### Sampling strategies

Sampling information to resample the data set.
When float, it corresponds to the desired ratio of the number of samples in the minority class over
the number of samples in the majority class after resampling.
When str, specify the class targeted by the resampling. The number of samples in the different classes
will be equalized. Possible choices are:

   - 'minority': resample only the minority class;
   - 'not minority': resample all classes but the minority class;
   - 'not majority': resample all classes but the majority class;
   - 'all': resample all classes;
   - 'auto': equivalent to 'not majority'.

When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
samples for each targeted class.
When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
The values correspond to the desired number of samples for each class.


```python
from deepmol.imbalanced_learn import SMOTEENN

d5 = deepcopy(data)
sampler = SMOTEENN(sampling_strategy=0.8, random_state=123, n_jobs=-1)
d5 = sampler.sample(d5)

Counter(d5.y)
```


    Counter({0: 922, 1: 772})



### SMOTETomek

SMOTETomek is a hybrid technique that combines two other techniques, SMOTE (Synthetic Minority Over-sampling Technique) and Tomek Links, to address the problem of imbalanced data in machine learning. SMOTE is used to oversample the minority class by creating synthetic samples, while Tomek Links is used to undersample the majority class by identifying and removing samples that are close to the boundary between the minority and majority classes.

#### Sampling strategies

Sampling information to resample the data set.
When float, it corresponds to the desired ratio of the number of samples in the minority class over
the number of samples in the majority class after resampling.
When str, specify the class targeted by the resampling. The number of samples in the different classes
will be equalized. Possible choices are:

   - 'minority': resample only the minority class;
   - 'not minority': resample all classes but the minority class;
   - 'not majority': resample all classes but the majority class;
   - 'all': resample all classes;
   - 'auto': equivalent to 'not majority'.

When dict, the keys correspond to the targeted classes. The values correspond to the desired number of
samples for each targeted class.
When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
The values correspond to the desired number of samples for each class.


```python
from deepmol.imbalanced_learn import SMOTETomek

d6 = deepcopy(data)
sampler = SMOTETomek(sampling_strategy=0.7, random_state=123, n_jobs=-1)
d6 = sampler.sample(d6)

Counter(d6.y)
```


    Counter({1: 693, 0: 993})


