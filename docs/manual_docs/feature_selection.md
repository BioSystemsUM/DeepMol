# Feature Selection

Regarding feature selection it is possible to do Low Variance Feature Selection, 
KBest, Percentile, Recursive Feature Elimination and selecting features based on 
importance weights.

```python
from deepmol.feature_selection import LowVarianceFS

# Feature Selection to remove features with low variance across molecules
LowVarianceFS(0.15).select_features(dataset)

# print shape of the dataset to see difference in the X shape (fewer features)
dataset.get_shape()

((1000,), (1000, 35), (1000,))
```