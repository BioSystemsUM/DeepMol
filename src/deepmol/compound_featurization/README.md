# How to implement a new featurization method in DeepMol

This is a brief description on how you can contribute or add more featurizers to your workflows. 

You have to extend the class **MolecularFeaturizer** 

```python
from deepmol.compound_featurization import MolecularFeaturizer

class MyFeaturizer(MolecularFeaturizer):

    def __init__(my_parameter, **kwargs):

        self.my_parameter = my_parameter

        self.features_names = [f'my_featurizer_{i}' for i in range(64)]
    
    def _featurize(self, mol: Mol) -> np.ndarray:
        # implement the featurization method here

```

