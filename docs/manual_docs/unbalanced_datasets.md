# Unbalanced Datasets

Multiple methods to deal with unbalanced datasets can be used to do oversampling,
under-sampling or a mixture of both (Random, SMOTE, SMOTEENN, SMOTETomek and 
ClusterCentroids).

```python
from deepmol.imbalanced_learn.imbalanced_learn import SMOTEENN

train_dataset = SMOTEENN().sample(train_dataset)
```