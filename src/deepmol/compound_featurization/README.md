# How to implement a new featurization method in DeepMol

This is a brief description on how you can contribute or add more featurizers to your workflows. 

Mandatory steps:
1. Create a new python file in the folder `deepmol/compound_featurization/` with the name of your featurizer.
2. Implement the class that extends the **MolecularFeaturizer** class.
3. Implement the method `_featurize` that will return the features of the molecule.
4. Implement the instance variable **features_names** that will contain the names of the features.

Optional steps:
1. Implement instance variables for parameters of the method like **my_parameter**.

```python
from deepmol.compound_featurization import MolecularFeaturizer
from rdkit.Chem import Mol
import numpy as np

class MyFeaturizer(MolecularFeaturizer):

    def __init__(self, my_parameter, **kwargs):
        super().__init__(**kwargs)
        
        self.my_parameter = my_parameter

        self.features_names = [f'my_featurizer_{i}' for i in range(64)]
    
    def _featurize(self, mol: Mol) -> np.ndarray:
        # implement the featurization method here

```

# Testing

Testing these classes is also mandatory. 

## Unit tests

You can create a new file in the folder `tests/unit_tests/featurizers/` with the name of 
your featurizer with "test_" as prefix. 

How to implement a unit test for featurizers?

Mandatory steps:
1. Create a new python file in the folder `tests/unit_tests/featurizers/` with the name of your 
featurizer with "test_" as prefix.
2. Implement the class that extends the **FeaturizerTestCase** class and **TestCase**.
3. Implement the method `test_featurize` that will test the featurize method of the featurizer. 
For this use the `self.mock_dataset` attribute. It is a pre-built dataset with 7 molecules for this kind of tests.
4. Implement the method `test_featurize_with_nan` that will test the featurize method of the featurizer 
with a dataset that contains invalid smiles. For this use the `self.mock_dataset_with_invalid` attribute. 
It is a pre-built dataset with 7 molecules (some are invalid) for this kind of tests.
5. You can implement more tests if you want.

Don't forget to import the necessary classes and methods and check the coverage of your tests.

Example:

```python
from copy import copy

from deepmol.compound_featurization.nc_mfp_generator import NcMfp
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
from unittest import TestCase

from rdkit.Chem import MolFromSmiles


class TestNcMfp(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        transformer = NcMfp()
        transformer.featurize(self.mock_dataset, inplace=True)
        self.assertEqual(7, self.mock_dataset._X.shape[0])
        self.assertEqual(254399, self.mock_dataset._X.shape[1])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        NcMfp().featurize(dataset, inplace=True)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

    def test_fingerprint_with_no_ones(self):
        transformer = NcMfp()
        bitstring = transformer._featurize(MolFromSmiles("COC1=CC(=CC(=C1OC)OC)CCN"))
        self.assertEqual(bitstring.shape[0], 254399)
```

## Integration tests

You can create a method in the file `tests/integration_tests/dataset/test_dataset_features.py` that will test 
your method. The main goal of this test is to check if the featurizer is working with the dataset class. 

Mandatory steps:
1. Create a new method in the class that extends the **TestDataset** class.
2. Implement the method that will test the featurize method of the featurizer using 
the self.small_dataset_to_test attribute, a dataset with 13 molecules already chosen and loaded to test the featurizers.


Example:

```python
from tests.integration_tests.dataset.test_dataset import TestDataset

class TestDatasetFeaturizers(TestDataset):
    
    ...
    def test_dataset_with_neural_npfp(self):
        from deepmol.compound_featurization import NeuralNPFP
        NeuralNPFP().featurize(self.small_dataset_to_test, inplace=True)
        self.assertEqual(self.small_dataset_to_test.X.shape[0], 13)
        self.assertEqual(self.small_dataset_to_test.X.shape[1], 64)

```