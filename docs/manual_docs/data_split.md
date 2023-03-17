# Data Split

Data can be split randomly or using stratified splitters. K-fold split, train-test
split and train-validation-test split can be used.

```python
from deepmol.splitters.splitters import SingletaskStratifiedSplitter

# Data Split
splitter = SingletaskStratifiedSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset, frac_train=0.7,
                                                                             frac_valid=0.15, frac_test=0.15)
train_dataset.get_shape()

((1628,), (1628, 1024), (1628,))

valid_dataset.get_shape()

((348,), (348, 1024), (348,))

test_dataset.get_shape()

((350,), (350, 1024), (350,))
```