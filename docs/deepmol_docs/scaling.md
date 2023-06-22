# Data Scaling with DeepMol

In machine learning and deep learning, scaling is important because it can significantly affect the performance of the algorithms. This is also true in the field of chemoinformatics, which involves the use of machine learning and other computational methods to analyze chemical data.

One reason why scaling is important is that many machine learning algorithms use distance-based measures to calculate similarities between data points. If the features of the data are not scaled, the algorithm may give more weight to features with larger values, even if they are not more important for the analysis. This can lead to biased results and suboptimal model performance.

Another reason why scaling is important is that it can help to speed up the training process. When the features of the data are not scaled, the optimization algorithm used in training may take longer to converge or may even fail to converge at all. This can be especially problematic in deep learning, where large amounts of data and complex models are often used.

As we will see below, DeepMol offers a wide variety of scalers.

<font size="5"> **Let's start by loading some data** </font>


```python
from deepmol.compound_featurization import TwoDimensionDescriptors
from deepmol.loaders import CSVLoader

# Load data from CSV file
loader = CSVLoader(dataset_path='../data/CHEMBL217_reduced.csv',
                   smiles_field='SMILES',
                   id_field='Original_Entry_ID',
                   labels_fields=['Activity_Flag'],
                   mode='auto',
                   shard_size=2500)
# create the dataset
data = loader.create_dataset(sep=',', header=0)
# create the features
TwoDimensionDescriptors().featurize(data, inplace=True)
```

```python
# view the features
data.X # data is very heterogeneous
```




    array([[ 1.2918277e+01,  1.7210670e-02,  1.2918277e+01, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 1.3306055e+01, -5.6191501e-03,  1.3306055e+01, ...,
             0.0000000e+00,  0.0000000e+00,  1.0000000e+00],
           [ 1.1760469e+01, -2.3133385e-01,  1.1760469e+01, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           ...,
           [ 1.2702195e+01, -3.6049552e+00,  1.2702195e+01, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 2.4115279e+00,  4.3334645e-01,  2.4115279e+00, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 6.1730037e+00,  6.9678569e-01,  6.1730037e+00, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00]], dtype=float32)



## StandardScaler

Standardize features by removing the mean and scaling to unit variance.


```python
from copy import deepcopy
from deepmol.scalers import StandardScaler

d1 = deepcopy(data)
scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
scaler.fit_transform(d1, inplace=True) # you can scale only a portion of the data by passing a columns argument with the indexes of the columns to scale
d1.X # the data is much more homogeneous
```




    array([[ 0.6015701 ,  0.5215607 ,  0.6015701 , ..., -0.2591845 ,
            -0.32493442, -0.24450661],
           [ 0.7382216 ,  0.5070287 ,  0.7382216 , ..., -0.2591845 ,
            -0.32493442,  4.0300846 ],
           [ 0.19356354,  0.36335236,  0.19356354, ..., -0.2591845 ,
            -0.32493442, -0.24450661],
           ...,
           [ 0.5254238 , -1.7840909 ,  0.5254238 , ..., -0.2591845 ,
            -0.32493442, -0.24450661],
           [-3.1009648 ,  0.78644764, -3.1009648 , ..., -0.2591845 ,
            -0.32493442, -0.24450661],
           [-1.7754363 ,  0.9541371 , -1.7754363 , ..., -0.2591845 ,
            -0.32493442, -0.24450661]], dtype=float32)



## MinMaxScaler

Transform features by scaling each feature to a given range.


```python
from deepmol.scalers import MinMaxScaler

d2 = deepcopy(data)
scaler = MinMaxScaler(feature_range=(-2, 2))
scaler.fit_transform(d2, inplace=True)
d2.X # data is scaled between -2 and 2
```




    array([[ 1.2589104 ,  1.5757873 ,  1.2589104 , ..., -2.        ,
            -2.        , -2.        ],
           [ 1.3791888 ,  1.567372  ,  1.3791888 , ..., -2.        ,
            -2.        ,  0.        ],
           [ 0.8997896 ,  1.4841713 ,  0.8997896 , ..., -2.        ,
            -2.        , -2.        ],
           ...,
           [ 1.1918876 ,  0.24062085,  1.1918876 , ..., -2.        ,
            -2.        , -2.        ],
           [-2.        ,  1.729179  , -2.        , ..., -2.        ,
            -2.        , -2.        ],
           [-0.83329153,  1.8262854 , -0.83329153, ..., -2.        ,
            -2.        , -2.        ]], dtype=float32)



## MaxAbsScaler

Scale each feature by its maximum absolute value.


```python
from deepmol.scalers import MaxAbsScaler

d3 = deepcopy(data)
scaler = MaxAbsScaler()
scaler.fit_transform(d3, inplace=True)
d3.X
```




    array([[ 8.4391510e-01,  1.7773149e-03,  8.4391510e-01, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 8.6924756e-01, -5.8027951e-04,  8.6924756e-01, ...,
             0.0000000e+00,  0.0000000e+00,  5.0000000e-01],
           [ 7.6827878e-01, -2.3889430e-02,  7.6827878e-01, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           ...,
           [ 8.2979906e-01, -3.7227723e-01,  8.2979906e-01, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 1.5753841e-01,  4.4750907e-02,  1.5753841e-01, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 4.0326515e-01,  7.1955800e-02,  4.0326515e-01, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00]], dtype=float32)



## RobustScaler

Scale features using statistics that are robust to outliers.


```python
from deepmol.scalers import RobustScaler

d4 = deepcopy(data)
scaler = RobustScaler()
scaler.fit_transform(d4, inplace=True)
d4.X # scaled data
```




    array([[ 0.22212751,  0.29477584,  0.22212751, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.3872069 ,  0.26957947,  0.3872069 , ...,  0.        ,
             0.        ,  1.        ],
           [-0.27075756,  0.02046694, -0.27075756, ...,  0.        ,
             0.        ,  0.        ],
           ...,
           [ 0.13014036, -3.7028677 ,  0.13014036, ...,  0.        ,
             0.        ,  0.        ],
           [-4.250654  ,  0.75404876, -4.250654  , ...,  0.        ,
             0.        ,  0.        ],
           [-2.649373  ,  1.0447963 , -2.649373  , ...,  0.        ,
             0.        ,  0.        ]], dtype=float32)



## Normalizer scaler

Normalize samples individually to unit norm.


```python
from deepmol.scalers import Normalizer

d5 = deepcopy(data)
scaler = Normalizer(norm='l2') # One of 'l1', 'l2' or 'max'. The norm to use to normalize each non-zero sample.
scaler.fit_transform(d5, inplace=True)
d5.X # scaled data
```




    array([[ 2.4346070e-07,  3.2435607e-10,  2.4346070e-07, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 3.5423795e-08, -1.4959476e-11,  3.5423795e-08, ...,
             0.0000000e+00,  0.0000000e+00,  2.6622313e-09],
           [ 7.4276683e-04, -1.4610566e-05,  7.4276683e-04, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           ...,
           [ 2.1571793e-06, -6.1221971e-07,  2.1571793e-06, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 7.5876460e-06,  1.3634839e-06,  7.5876460e-06, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
           [ 1.9041399e-05,  2.1493222e-06,  1.9041399e-05, ...,
             0.0000000e+00,  0.0000000e+00,  0.0000000e+00]], dtype=float32)



## Binarizer scaler

Binarize data (set feature values to 0 or 1) according to a threshold.


```python
from deepmol.scalers import Binarizer

d6 = deepcopy(data)
scaler = Binarizer(threshold=1) # features higher than 10 are set to 1, features lower than 10 are set to 0
scaler.fit_transform(d6, inplace=True)
d6.X
```




    array([[1., 0., 1., ..., 0., 0., 0.],
           [1., 0., 1., ..., 0., 0., 0.],
           [1., 0., 1., ..., 0., 0., 0.],
           ...,
           [1., 0., 1., ..., 0., 0., 0.],
           [1., 0., 1., ..., 0., 0., 0.],
           [1., 0., 1., ..., 0., 0., 0.]], dtype=float32)



## QuantileTransformer

The QuantileTransformer is a preprocessing method that transforms input data to have a specified probability distribution. This function maps the data to a uniform or normal distribution using the quantiles of the input data.

This transformer is often useful when working with machine learning algorithms that are sensitive to the scale and distribution of the input data, such as neural networks. The QuantileTransformer is particularly useful when the input data has a highly skewed distribution, as it can transform the data to a more Gaussian distribution.


```python
from deepmol.scalers import QuantileTransformer

d7 = deepcopy(data)
scaler = QuantileTransformer()
scaler.fit_transform(d7, inplace=True)
d7.X # scale data
```




    array([[0.72686756, 0.7179994 , 0.72686756, ..., 0.        , 0.        ,
            0.        ],
           [0.86711156, 0.7053635 , 0.86711156, ..., 0.        , 0.        ,
            0.9714715 ],
           [0.32533428, 0.51655453, 0.32533428, ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.62756836, 0.12748909, 0.62756836, ..., 0.        , 0.        ,
            0.        ],
           [0.        , 0.8474799 , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.14120819, 0.92178524, 0.14120819, ..., 0.        , 0.        ,
            0.        ]], dtype=float32)



## PowerTransformer

The PowerTransformer is a preprocessing function in scikit-learn that applies a power transformation to make the data more Gaussian-like. It can be used for data with skewed distributions, as well as data that has a linear relationship with the target variable.

The PowerTransformer applies either a Box-Cox transformation or a Yeo-Johnson transformation to the input data. The Box-Cox transformation is only applicable to positive data, while the Yeo-Johnson transformation can be applied to both positive and negative data.


```python
from deepmol.scalers import PowerTransformer

d8 = deepcopy(data)
scaler = PowerTransformer(method='yeo-johnson', # The power transform method. Available methods are: 'yeo-johnson', works with positive and negative values; box-cox', only works with strictly positive values
                          standardize=True) # apply zero mean, unit variance normalization to the transformed output
scaler.fit_transform(d8, inplace=True)
d8.X # scaled data
```





    array([[ 0.6452927 ,  0.36367184,  0.6452927 , ..., -0.26236013,
            -0.4512646 , -0.24539872],
           [ 0.93890953,  0.33411208,  0.93890953, ..., -0.26236013,
            -0.4512646 ,  4.075001  ],
           [-0.0993756 ,  0.07504444, -0.0993756 , ..., -0.26236013,
            -0.4512646 , -0.24539872],
           ...,
           [ 0.49173132, -1.5438824 ,  0.49173132, ..., -0.26236013,
            -0.4512646 , -0.24539872],
           [-1.921834  ,  1.0273771 , -1.921834  , ..., -0.26236013,
            -0.4512646 , -0.24539872],
           [-1.7411773 ,  1.5712001 , -1.7411773 , ..., -0.26236013,
            -0.4512646 , -0.24539872]], dtype=float32)


