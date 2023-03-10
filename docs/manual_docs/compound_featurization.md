# Compound Featurization

It is possible to compute multiple types of molecular fingerprints like Morgan
Fingerprints, MACCS Keys, Layered Fingerprints, RDK Fingerprints and AtomPair 
Fingerprints. Featurizers from DeepChem and molecular embeddings like the 
Mol2Vec can also be computed. More complex molecular embeddings like the 
Seq2Seq and transformer-based are in  development and will be added soon.

```python
from deepmol.compound_featurization import MorganFingerprint

# Compute morgan fingerprints for molecules in the previous loaded dataset
MorganFingerprint(radius=2, size=1024).featurize(dataset)
# view the computed features (dataset.X)
dataset.X
#print shape of the dataset to see difference in the X shape
dataset.get_shape()

((1000,), (1000, 1024), (1000,))
```